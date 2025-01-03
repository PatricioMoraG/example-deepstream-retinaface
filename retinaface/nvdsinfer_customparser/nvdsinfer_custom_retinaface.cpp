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

// Función para aplicar Non-Maximum Suppression (NMS)
// Nota: Implementa esta función según tus necesidades
// Aquí se proporciona una implementación básica como ejemplo
std::vector<RetinaFaceDetection> applyNMS(const std::vector<RetinaFaceDetection>& detections, float nmsThreshold) {
    std::vector<RetinaFaceDetection> nmsDetections;
    // Ordenar las detecciones por confianza descendente
    std::vector<RetinaFaceDetection> sortedDetections = detections;
    std::sort(sortedDetections.begin(), sortedDetections.end(),
              [](const RetinaFaceDetection& a, const RetinaFaceDetection& b) -> bool {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sortedDetections.size(), false);

    for (size_t i = 0; i < sortedDetections.size(); ++i) {
        if (suppressed[i])
            continue;

        nmsDetections.push_back(sortedDetections[i]);

        for (size_t j = i + 1; j < sortedDetections.size(); ++j) {
            if (suppressed[j])
                continue;

            // Calcular IoU entre sortedDetections[i] y sortedDetections[j]
            float x1 = std::max(sortedDetections[i].x1, sortedDetections[j].x1);
            float y1 = std::max(sortedDetections[i].y1, sortedDetections[j].y1);
            float x2 = std::min(sortedDetections[i].x2, sortedDetections[j].x2);
            float y2 = std::min(sortedDetections[i].y2, sortedDetections[j].y2);

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float areaA = (sortedDetections[i].x2 - sortedDetections[i].x1) * (sortedDetections[i].y2 - sortedDetections[i].y1);
            float areaB = (sortedDetections[j].x2 - sortedDetections[j].x1) * (sortedDetections[j].y2 - sortedDetections[j].y1);
            float iou = intersection / (areaA + areaB - intersection);

            if (iou > nmsThreshold)
                suppressed[j] = true;
        }
    }

    return nmsDetections;
}

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

    // Índices para recorrer los buffers loc, landm, conf
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
        // Asumimos que la red te ha devuelto:
        //
        // locData  con shape: [ (4 * anchorCount) * featSize ]
        // landmData con shape:[ (10 * anchorCount) * featSize ]
        // confData con shape: [ (2 * anchorCount) * featSize ]
        //
        // por lo tanto, cada celda en locData ocupa 4 * anchorCount floats
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
                    // confData en offset = (2 * anchorCount) * cellIndex + (k * 2)
                    //   0 => bg, 1 => face
                    float c1 = confData[confOffset + (2 * anchorCount) * cellIndex + (k * 2) + 0];
                    float c2 = confData[confOffset + (2 * anchorCount) * cellIndex + (k * 2) + 1];

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
                    // locData en offset = (4 * anchorCount) * cellIndex + (k * 4)
                    float dx = locData[locOffset + (4 * anchorCount) * cellIndex + (k * 4) + 0];
                    float dy = locData[locOffset + (4 * anchorCount) * cellIndex + (k * 4) + 1];
                    float dw = locData[locOffset + (4 * anchorCount) * cellIndex + (k * 4) + 2];
                    float dh = locData[locOffset + (4 * anchorCount) * cellIndex + (k * 4) + 3];

                    // Center de la celda (en normalizado)
                    float prior_cx = (x + 0.5f) / feat_w;
                    float prior_cy = (y + 0.5f) / feat_h;

                    // Ancho y alto del anchor (normalizado)
                    float prior_w  = (anchorSize * (k + 1)) / static_cast<float>(inputWidth);
                    float prior_h  = (anchorSize * (k + 1)) / static_cast<float>(inputHeight);

                    // Aplicar escala del decode.cu
                    //   cx = prior_cx + dx * 0.1 * prior_w
                    //   cy = prior_cy + dy * 0.1 * prior_h
                    //   w  = prior_w  * exp(dw * 0.2)
                    //   h  = prior_h  * exp(dh * 0.2)
                    float cx = prior_cx + dx * 0.1f * prior_w;
                    float cy = prior_cy + dy * 0.1f * prior_h;
                    float w  = prior_w  * std::exp(dw * 0.2f);
                    float h  = prior_h  * std::exp(dh * 0.2f);

                    // Convertir de (cx, cy, w, h) a (x1, y1, x2, y2) en píxeles
                    float x1 = (cx - w * 0.5f) * inputWidth;
                    float y1 = (cy - h * 0.5f) * inputHeight;
                    float x2 = (cx + w * 0.5f) * inputWidth;
                    float y2 = (cy + h * 0.5f) * inputHeight;

                    // -------------------
                    // Extraer Landmarks
                    // -------------------
                    // landmData offset = (10 * anchorCount) * cellIndex + (k * 10)
                    float lm[10];
                    for (int m = 0; m < 5; m++)
                    {
                        float ldx = landmData[landmOffset + (10 * anchorCount) * cellIndex + (k * 10) + (2 * m + 0)];
                        float ldy = landmData[landmOffset + (10 * anchorCount) * cellIndex + (k * 10) + (2 * m + 1)];

                        // Decodificar
                        float lx = prior_cx + ldx * 0.1f * prior_w;
                        float ly = prior_cy + ldy * 0.1f * prior_h;

                        // Pasar a píxeles
                        lx *= inputWidth;
                        ly *= inputHeight;
                        lm[2 * m + 0] = lx;
                        lm[2 * m + 1] = ly;
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
                    for (int m = 0; m < 10; ++m)
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
    // float nmsThreshold = 0.4f; // Define un umbral adecuado
    // detections = applyNMS(detections, nmsThreshold);

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

    // 2) Obtener dimensiones de entrada de la red
    //    networkInfo.width = ancho, networkInfo.height = alto
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

    // Obtener umbral de confianza y NMS del detectionParams
    float confThreshold = detectionParams.per_class_threshold[0]; // Asumiendo una sola clase
    float nmsThreshold = detectionParams.nms_threshold;

    // Manejar batchSize
    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        // Calcular punteros para el batch actual
        const float* locBatchData   = locData + batchIdx * numBboxes * 4;
        const float* landmBatchData = landmData + batchIdx * numBboxes * 10;
        const float* confBatchData  = confData + batchIdx * numBboxes * 2;

        // Decodificar las detecciones para el batch actual
        auto dets = decodeRetinaFace(locBatchData, landmBatchData, confBatchData, inputW, inputH, confThreshold);

        // Aplicar NMS si es necesario
        dets = applyNMS(dets, nmsThreshold);

        // Recorrer las detecciones después de NMS
        for (auto& det : dets) {
            float score = det.confidence;

            // Filtrar por umbral de confianza (redundante si ya se filtró en decodeRetinaFace)
            if (score < confThreshold) {
                continue;
            }

            // Obtener coordenadas
            float x1 = det.x1;
            float y1 = det.y1;
            float x2 = det.x2;
            float y2 = det.y2;

            // Descartar cajas degeneradas
            if ((x2 - x1) < 1 || (y2 - y1) < 1) {
                continue;
            }

            // Agregar el objeto (rostro) a la lista
            NvDsInferObjectDetectionInfo objInfo;
            objInfo.classId = 0;  // "0" = rostro (asumiendo que es la única clase)
            objInfo.detectionConfidence = score;
            objInfo.left   = x1;
            objInfo.top    = y1;
            objInfo.width  = (x2 - x1);
            objInfo.height = (y2 - y1);

            objectList.push_back(objInfo);

            // Si deseas agregar atributos adicionales (como landmarks), puedes hacerlo aquí
            // Por ejemplo:
            /*
            NvDsInferAttribute attr;
            attr.attributeIndex = 0; // Define según corresponda
            attr.attributeValue = score;
            attrList.push_back(attr);
            */
        }
    }

    return true;
}
