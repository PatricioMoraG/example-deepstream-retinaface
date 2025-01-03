/*******************************************************************************
 * Custom parser para RetinaFace en DeepStream (versión corregida)
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "nvdsinfer_custom_retinaface.h"
// #include <nvinfer.h>  // (opcional, sólo si necesitas APIs directas de TRT)

// --------------------------------------------------------------------------------
// Función para generar anchors (priors) equivalente al Python PriorBox
// --------------------------------------------------------------------------------
std::vector<float> generate_retinaface_anchors(
    const std::vector<std::vector<int>>& min_sizes,
    const std::vector<int>& steps,
    int input_height,
    int input_width,
    bool clip)
{
    std::vector<std::pair<int,int>> feature_maps;
    feature_maps.reserve(steps.size());

    // Calcular (ceil(H/step), ceil(W/step)) para cada escala
    for (auto step : steps) {
        int fm_h = static_cast<int>(std::ceil(static_cast<float>(input_height) / step));
        int fm_w = static_cast<int>(std::ceil(static_cast<float>(input_width)  / step));
        feature_maps.emplace_back(fm_h, fm_w);
    }

    // Generar anchors en [cx, cy, w, h]
    std::vector<float> anchors;
    anchors.reserve(100000); // Evita realocaciones si esperas ~16800 anchors

    for (size_t k = 0; k < feature_maps.size(); ++k) {
        int fm_h = feature_maps[k].first;
        int fm_w = feature_maps[k].second;

        for (int i = 0; i < fm_h; ++i) {
            for (int j = 0; j < fm_w; ++j) {
                for (auto min_size : min_sizes[k]) {
                    // w,h normalizados a [0,1]
                    float s_kx = static_cast<float>(min_size) / input_width;
                    float s_ky = static_cast<float>(min_size) / input_height;

                    // centro x,y normalizado
                    float cx = ((j + 0.5f) * steps[k]) / static_cast<float>(input_width);
                    float cy = ((i + 0.5f) * steps[k]) / static_cast<float>(input_height);

                    anchors.push_back(cx);
                    anchors.push_back(cy);
                    anchors.push_back(s_kx);
                    anchors.push_back(s_ky);
                }
            }
        }
    }

    // (Opcional) Recortar a [0,1]
    if (clip) {
        for (auto &v : anchors) {
            v = std::min(std::max(v, 0.0f), 1.0f);
        }
    }

    return anchors;
}

// --------------------------------------------------------------------------------
// Config "estática" equivalente a cfg_re50 (Python)
// --------------------------------------------------------------------------------
static const std::vector<std::vector<int>> kMinSizes = {
    {16, 32}, {64, 128}, {256, 512}
};
static const std::vector<int> kSteps = {8, 16, 32};

// Vector global donde guardamos los priors una sola vez.
static std::vector<float> myPriorAnchors;

// --------------------------------------------------------------------------------
// Estructuras auxiliares
// --------------------------------------------------------------------------------
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} BBox;

// Variances típicas usadas en RetinaFace
static const float VARIANCE[2] = {0.1f, 0.2f};

// --------------------------------------------------------------------------------
// Decodificar cada bbox (loc) usando priors y variances
// --------------------------------------------------------------------------------
static BBox decodeBBox(const float loc[4], const float prior[4])
{
    // prior: [cx, cy, w, h]
    // loc:   [dx, dy, dw, dh]
    BBox box;

    float cx = prior[0] + loc[0] * VARIANCE[0] * prior[2];
    float cy = prior[1] + loc[1] * VARIANCE[0] * prior[3];
    float w  = prior[2] * std::exp(loc[2] * VARIANCE[1]);
    float h  = prior[3] * std::exp(loc[3] * VARIANCE[1]);

    // (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
    box.x1 = cx - w * 0.5f;
    box.y1 = cy - h * 0.5f;
    box.x2 = cx + w * 0.5f;
    box.y2 = cy + h * 0.5f;

    return box;
}

// --------------------------------------------------------------------------------
// Función principal para parsear las salidas de RetinaFace en DeepStream
// --------------------------------------------------------------------------------

extern "C"
bool NvDsInferParseCustomRetinaFace(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList,
    std::vector<NvDsInferAttribute> &attrList,
    void* customData,
    int batchSize)
{
    // 1) Validar que tengamos al menos 3 salidas (loc, landm, conf)
    if (outputLayersInfo.size() < 3) {
        std::cerr << "ERROR: Se esperan al menos 3 salidas [loc, landms, conf]." << std::endl;
        return false;
    }

    // 2) Generar (o usar) priors si aún no los tenemos
    //    OJO: networkInfo.width = ancho, networkInfo.height = alto
    int inputW = networkInfo.width;
    int inputH = networkInfo.height;

    if (myPriorAnchors.empty()) {
        myPriorAnchors = generate_retinaface_anchors(kMinSizes, kSteps, inputH, inputW, false);
        // Con 'clip = true' si tu modelo lo requiere
    }
    if (myPriorAnchors.empty()) {
        std::cerr << "ERROR: no se pudieron generar priors" << std::endl;
        return false;
    }

    // 3) Obtener punteros a las 3 salidas
    const NvDsInferLayerInfo &locLayer   = outputLayersInfo[0];
    //const NvDsInferLayerInfo &landmLayer = outputLayersInfo[1]; // si luego parseas landmarks
    const NvDsInferLayerInfo &confLayer  = outputLayersInfo[2];

    // Convierte buffer a float*
    const float* locData  = reinterpret_cast<const float*>(locLayer.buffer);
    // const float* landmData = reinterpret_cast<const float*>(landmLayer.buffer); // Solo si usarás landmarks
    const float* confData = reinterpret_cast<const float*>(confLayer.buffer);

    // 4) Calcular cuántas detecciones (N=16800, etc.)
    //    locLayer.inferDims = (16800, 4) => inferDims.d[0] = 16800
    size_t numBboxes = (locLayer.inferDims.numDims > 0) ? locLayer.inferDims.d[0] : 0;
    if (numBboxes == 0) {
        std::cerr << "ERROR: locLayer.inferDims es inesperado (0)." << std::endl;
        return false;
    }

    // 5) Determinar si usamos 'customData' o si usamos 'myPriorAnchors'
    //    (Ejemplo: si 'customData' != nullptr, lo tomamos como priors externOs, 
    //     de lo contrario usamos 'myPriorAnchors')
    const float* priorData = nullptr;
    if (customData) {
        priorData = reinterpret_cast<const float*>(customData);
    } else {
        // Sin 'customData', usamos nuestros priors globales
        priorData = myPriorAnchors.data();
    }

    // 6) Parsear y decodificar
    objectList.clear();
    objectList.reserve(numBboxes);

    // Umbral de confianza (asumiendo 1 sola clase, "rostro")
    float detectionThreshold = detectionParams.perClassThreshold[0];

    for (size_t i = 0; i < numBboxes; ++i) {
        // 6.1) Confianza de la clase "rostro" en canal 1
        float score = confData[i * 2 + 1];
        if (score < detectionThreshold) {
            continue;
        }

        // 6.2) Decodificar la bbox
        const float* locPtr   = &locData[i * 4];
        const float* priorPtr = &priorData[i * 4];

        BBox box = decodeBBox(locPtr, priorPtr);

        // 6.3) Convertir [0,1] -> píxeles absolutos
        float x1 = box.x1 * inputW;
        float y1 = box.y1 * inputH;
        float x2 = box.x2 * inputW;
        float y2 = box.y2 * inputH;

        // 6.4) Clip a [0, ancho/alto]
        x1 = std::max(0.0f, std::min(x1, (float)inputW - 1));
        y1 = std::max(0.0f, std::min(y1, (float)inputH - 1));
        x2 = std::max(0.0f, std::min(x2, (float)inputW - 1));
        y2 = std::max(0.0f, std::min(y2, (float)inputH - 1));

        // Descartar boxes degeneradas
        if ((x2 - x1) < 1 || (y2 - y1) < 1) {
            continue;
        }

        // 6.5) Agregamos el objeto (rostro) a la lista
        NvDsInferObjectDetectionInfo objInfo;
        objInfo.classId = 0;  // "0" = rostro (asumiendo que es la única clase)
        objInfo.detectionConfidence = score;
        objInfo.left = x1;
        objInfo.top  = y1;
        objInfo.width  = (x2 - x1);
        objInfo.height = (y2 - y1);

        objectList.push_back(objInfo);

        // (Opcional) Parsear landmarks aquí si lo deseas.
        // const float* landmPtr = &landmData[i * 10];
        // ...
    }

    // 7) Retornar verdadero si todo fue OK
    return true;
}
