/*******************************************************************************
 * Custom parser para RetinaFace en DeepStream (versión corregida)
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "nvdsinfer_custom_retinaface.h"

// Estructura auxiliar para stride/anchor
struct StrideAnchor {
    int stride;
    int baseAnchor;
};

// Ejemplo de anclas para los 3 niveles, tal como en decode.cu
static const StrideAnchor kStrideAnchors[3] = {
    {8,   16},
    {16,  64},
    {32,  256},
};

std::vector<RetinaFaceDetection> decodeRetinaFace(
    const float* locData,
    const float* landmData,
    const float* confData,
    int inputWidth,      // INPUT_W
    int inputHeight,     // INPUT_H
    float confThreshold  // por ej. 0.5 o 0.6
)
{
    std::vector<RetinaFaceDetection> detections;

    // Indices para ir recorriendo los buffers loc, landm, conf
    // ya que están concatenados por escalas, pero en buffers separados.
    int locOffset   = 0;
    int landmOffset = 0;
    int confOffset  = 0;

    // Recorremos los 3 niveles de FPN
    for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx)
    {
        const int stride     = kStrideAnchors[scaleIdx].stride;    // 8,16,32
        const int anchorSize = kStrideAnchors[scaleIdx].baseAnchor; // 16,64,256

        // Cuántas celdas hay en cada dimensión
        const int feat_w = inputWidth  / stride;
        const int feat_h = inputHeight / stride;
        const int featSize = feat_w * feat_h;

        // Procesamos 2 anchors por posición (k=0..1)
        const int anchorCount = 2; 

        // Para cada celda (feat_h * feat_w) y cada anchor, tenemos:
        // 4 floats de loc, 10 floats de landm, 2 floats de conf.
        // Sin embargo, en muchos modelos de TensorRT, se almacenan
        // en forma intercalada o en “channel major”. Para simplificar,
        // asumimos que la red te ha devuelto:
        //
        // locData  con shape: [ (4 * anchorCount) * featSize ]
        // landmData con shape:[ (10* anchorCount) * featSize ]
        // confData con shape: [ (2 * anchorCount) * featSize ]
        //
        // por lo tanto, cada celda en locData ocupa 4*anchorCount floats
        // y su index (x,y,k) lo calculamos manualmente.

        for (int y = 0; y < feat_h; ++y)
        {
            for (int x = 0; x < feat_w; ++x)
            {
                const int cellIndex = y * feat_w + x;

                for (int k = 0; k < anchorCount; ++k)
                {
                    // -------------------
                    // Extraer la confianza
                    // -------------------
                    // confData en offset = (2 * anchorCount)*cellIndex + (k * 2)
                    //   0 => bg, 1 => face
                    float c1 = confData[confOffset + (2 * anchorCount)*cellIndex + (k * 2) + 0];
                    float c2 = confData[confOffset + (2 * anchorCount)*cellIndex + (k * 2) + 1];

                    // Convertir (bg, face) en prob de "face" si se usa softmax a 2 clases
                    //   prob(face) = exp(c2) / (exp(c1) + exp(c2))
                    // O a veces la red ya te da (bg, face) en forma "logits" o "confidence direct".
                    // De acuerdo al decode.cu, se está usando:
                    //     conf2 = expf(conf2) / (expf(conf1) + expf(conf2))
                    float scoreFace = std::exp(c2) / (std::exp(c1) + std::exp(c2));

                    // Si no pasa umbral, saltamos
                    if (scoreFace < confThreshold)
                        continue;

                    // -------------------
                    // Extraer bbox
                    // -------------------
                    // locData en offset = (4 * anchorCount)*cellIndex + (k * 4)
                    float dx = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 0];
                    float dy = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 1];
                    float dw = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 2];
                    float dh = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 3];

                    // center de la celda (en normalizado)
                    float prior_cx = (x + 0.5f) / feat_w;
                    float prior_cy = (y + 0.5f) / feat_h;

                    // ancho y alto del anchor (normalizado)
                    float prior_w  = (anchorSize * (k + 1)) / (float)inputWidth;
                    float prior_h  = (anchorSize * (k + 1)) / (float)inputHeight;

                    // Aplicar scale del decode.cu
                    //   x = prior_cx + dx * 0.1 * prior_w
                    //   y = prior_cy + dy * 0.1 * prior_h
                    //   w = prior_w  * exp(dw * 0.2)
                    //   h = prior_h  * exp(dh * 0.2)
                    float cx = prior_cx + dx * 0.1f * prior_w;
                    float cy = prior_cy + dy * 0.1f * prior_h;
                    float w  = prior_w  * std::exp(dw * 0.2f);
                    float h  = prior_h  * std::exp(dh * 0.2f);

                    // Convertir de (cx, cy, w, h) a (x1, y1, x2, y2) normalizado
                    float x1 = cx - w * 0.5f;
                    float y1 = cy - h * 0.5f;
                    float x2 = cx + w * 0.5f;
                    float y2 = cy + h * 0.5f;

                    // Escalar a píxeles
                    x1 *= inputWidth;
                    y1 *= inputHeight;
                    x2 *= inputWidth;
                    y2 *= inputHeight;

                    // -------------------
                    // Extraer Landmarks
                    // -------------------
                    // landmData offset = (10 * anchorCount)*cellIndex + (k * 10)
                    float lm[10];
                    for (int m = 0; m < 5; m++)
                    {
                        float ldx = landmData[landmOffset + (10 * anchorCount)*cellIndex + (k * 10) + (2*m + 0)];
                        float ldy = landmData[landmOffset + (10 * anchorCount)*cellIndex + (k * 10) + (2*m + 1)];

                        // decodificar
                        float lx = prior_cx + ldx * 0.1f * prior_w;
                        float ly = prior_cy + ldy * 0.1f * prior_h;

                        // pasar a pixeles
                        lx *= inputWidth;
                        ly *= inputHeight;
                        lm[2*m + 0] = lx;
                        lm[2*m + 1] = ly;
                    }

                    // -------------------
                    // Guardar la detección
                    // -------------------
                    RetinaFaceDetection det;
                    det.x1 = x1;
                    det.y1 = y1;
                    det.x2 = x2;
                    det.y2 = y2;
                    det.confidence = scoreFace;
                    for (int m=0; m<10; ++m)
                        det.landmarks[m] = lm[m];

                    detections.push_back(det);
                } // k
            } // x
        } // y

        // Actualizamos offsets para el siguiente scale
        locOffset   += (4 * anchorCount) * featSize;
        landmOffset += (10 * anchorCount) * featSize;
        confOffset  += (2 * anchorCount) * featSize;
    }

    // En este punto, "detections" contiene TODAS las detecciones con confidence >= confThreshold
    // Se recomienda aplicar NMS para eliminar solapamientos.
    // Por ejemplo:
    // applyNMS(detections, nmsThreshold);

    return detections;
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

    // 3) Obtener punteros a las 3 salidas
    const NvDsInferLayerInfo &locLayer   = outputLayersInfo[0];
    const NvDsInferLayerInfo &landmLayer = outputLayersInfo[1]; 
    const NvDsInferLayerInfo &confLayer  = outputLayersInfo[2];

    // Convierte buffer a float*
    const float* locData  = reinterpret_cast<const float*>(locLayer.buffer);
    const float* landmData = reinterpret_cast<const float*>(landmLayer.buffer); 
    const float* confData = reinterpret_cast<const float*>(confLayer.buffer);

    // 4) Calcular cuántas detecciones (N=16800, etc.)
    //    locLayer.inferDims = (16800, 4) => inferDims.d[0] = 16800
    size_t numBboxes = (locLayer.inferDims.numDims > 0) ? locLayer.inferDims.d[0] : 0;
    if (numBboxes == 0) {
        std::cerr << "ERROR: locLayer.inferDims es inesperado (0)." << std::endl;
        return false;
    }

    // Suponiendo inputWidth=640, inputHeight=640, y confThreshold=0.5
    auto dets = decodeRetinaFace(locData, landmData, confData, 640, 640, 0.5f);

    //auto detsNMS = nmsRetinaFace(dets, nmsThreshold);

    std::vector<NvDsInferObjectDetectionInfo> objectList;
    for (auto& det : dets) {
        float score = det.score;

        // Filtramos por un umbral de confianza (si no lo hiciste ya en decodeRetinaFace)
        if (score < confThreshold) {
            continue;
        }

        // Obtenemos coordenadas
        float x1 = det.x1;
        float y1 = det.y1;
        float x2 = det.x2;
        float y2 = det.y2;

        // Descartar boxes degeneradas
        if ((x2 - x1) < 1 || (y2 - y1) < 1) {
            continue;
        }

        // 6.5) Agregamos el objeto (rostro) a la lista
        NvDsInferObjectDetectionInfo objInfo;
        objInfo.classId = 0;  // "0" = rostro (asumiendo que es la única clase)
        objInfo.detectionConfidence = score;
        objInfo.left   = x1;
        objInfo.top    = y1;
        objInfo.width  = (x2 - x1);
        objInfo.height = (y2 - y1);

        objectList.push_back(objInfo);
    }

    return true;
}
