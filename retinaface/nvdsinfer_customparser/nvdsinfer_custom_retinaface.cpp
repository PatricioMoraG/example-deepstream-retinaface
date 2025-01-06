/******************************************************************************
 * nvdsinfer_custom_retinaface.cpp
 * 
 * Implementación del parser personalizado de RetinaFace para DeepStream
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

// Incluye nuestro header con las declaraciones
#include "nvdsinfer_custom_retinaface.h"

//-------------------------------------------------------------------------------
// Anclas para 3 niveles de FPN, tal como en decode.cu (solo si el modelo usa 3 escalas)
//-------------------------------------------------------------------------------
static const StrideAnchor kStrideAnchors[3] = {
    { 8,  16},
    {16,  64},
    {32, 256}
};

//-------------------------------------------------------------------------------
// Implementación de decodeRetinaFace
//-------------------------------------------------------------------------------
std::vector<RetinaFaceDetection> decodeRetinaFace(
    const float* locData,
    const float* landmData,
    const float* confData,
    int inputWidth,
    int inputHeight,
    float confThreshold
)
{
    std::vector<RetinaFaceDetection> detections;

    int locOffset   = 0;
    int landmOffset = 0;
    int confOffset  = 0;

    // Se asume que la salida está separada por escalas (FPN) y que 
    // se iteran 3 escalas (kStrideAnchors).
    for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx)
    {
        const int stride     = kStrideAnchors[scaleIdx].stride;
        const int anchorSize = kStrideAnchors[scaleIdx].baseAnchor;

        const int feat_w = inputWidth  / stride;
        const int feat_h = inputHeight / stride;
        const int featSize = feat_w * feat_h;

        // Asumimos 2 anchors por celda
        const int anchorCount = 2; 

        for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {

                const int cellIndex = y * feat_w + x;

                for (int k = 0; k < anchorCount; ++k) {

                    // 1) Confianza de la cara
                    float c1 = confData[confOffset + (2 * anchorCount)*cellIndex + (k * 2) + 0]; // bg
                    float c2 = confData[confOffset + (2 * anchorCount)*cellIndex + (k * 2) + 1]; // face

                    float scoreFace = std::exp(c2) / (std::exp(c1) + std::exp(c2));
                    if (scoreFace < confThreshold) {
                        continue;
                    }

                    // 2) BBox
                    float dx = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 0];
                    float dy = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 1];
                    float dw = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 2];
                    float dh = locData[locOffset + (4 * anchorCount)*cellIndex + (k * 4) + 3];

                    float prior_cx = (x + 0.5f) / feat_w;
                    float prior_cy = (y + 0.5f) / feat_h;

                    float prior_w  = (anchorSize * (k + 1)) / static_cast<float>(inputWidth);
                    float prior_h  = (anchorSize * (k + 1)) / static_cast<float>(inputHeight);

                    float cx = prior_cx + dx * 0.1f * prior_w;
                    float cy = prior_cy + dy * 0.1f * prior_h;
                    float w  = prior_w  * std::exp(dw * 0.2f);
                    float h  = prior_h  * std::exp(dh * 0.2f);

                    float x1 = (cx - 0.5f * w) * inputWidth;
                    float y1 = (cy - 0.5f * h) * inputHeight;
                    float x2 = (cx + 0.5f * w) * inputWidth;
                    float y2 = (cy + 0.5f * h) * inputHeight;

                    // 3) Landmarks
                    float lm[10];
                    for (int m = 0; m < 5; ++m) {
                        float ldx = landmData[landmOffset + (10 * anchorCount)*cellIndex + (k * 10) + (2*m + 0)];
                        float ldy = landmData[landmOffset + (10 * anchorCount)*cellIndex + (k * 10) + (2*m + 1)];

                        float lx = prior_cx + ldx * 0.1f * prior_w;
                        float ly = prior_cy + ldy * 0.1f * prior_h;

                        lx *= inputWidth;
                        ly *= inputHeight;

                        lm[2*m + 0] = lx;
                        lm[2*m + 1] = ly;
                    }

                    // 4) Crear detección
                    RetinaFaceDetection det;
                    det.x1 = x1;
                    det.y1 = y1;
                    det.x2 = x2;
                    det.y2 = y2;
                    det.confidence = scoreFace;
                    for (int m = 0; m < 10; ++m) {
                        det.landmarks[m] = lm[m];
                    }

                    detections.push_back(det);
                }
            }
        }

        // Avanzar offsets para la siguiente escala
        locOffset   += (4 * anchorCount)  * featSize;
        landmOffset += (10 * anchorCount) * featSize;
        confOffset  += (2 * anchorCount)  * featSize;
    }

    return detections;
}

//-------------------------------------------------------------------------------
// Ejemplo de implementación de NMS (opcional). Ajusta según tus necesidades.
//-------------------------------------------------------------------------------
static std::vector<RetinaFaceDetection> applyNMS(
    const std::vector<RetinaFaceDetection> &dets, float nmsThreshold)
{
    if (dets.empty()) return {};

    // Ordenar por confianza descendente
    std::vector<RetinaFaceDetection> sorted = dets;
    std::sort(sorted.begin(), sorted.end(), 
              [](const auto &a, const auto &b){
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<RetinaFaceDetection> results;

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;

        results.push_back(sorted[i]);
        const auto &detA = sorted[i];
        float areaA = (detA.x2 - detA.x1) * (detA.y2 - detA.y1);

        // Comparar con detecciones siguientes
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;

            const auto &detB = sorted[j];
            float areaB = (detB.x2 - detB.x1) * (detB.y2 - detB.y1);

            float interX1 = std::max(detA.x1, detB.x1);
            float interY1 = std::max(detA.y1, detB.y1);
            float interX2 = std::min(detA.x2, detB.x2);
            float interY2 = std::min(detA.y2, detB.y2);

            float w = std::max(0.0f, interX2 - interX1);
            float h = std::max(0.0f, interY2 - interY1);
            float intersection = w * h;

            float iou = intersection / (areaA + areaB - intersection);
            if (iou > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

//-------------------------------------------------------------------------------
// Parser que DeepStream llama para extraer detecciones finales
//-------------------------------------------------------------------------------
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
    // Validar que tengamos al menos 3 salidas (loc, landm, conf)
    if (outputLayersInfo.size() < 3) {
        std::cerr << "ERROR: Se esperan al menos 3 salidas: loc, landms, conf." << std::endl;
        return false;
    }

    int inputW = networkInfo.width;
    int inputH = networkInfo.height;

    // Punteros a las salidas
    const NvDsInferLayerInfo &locLayer   = outputLayersInfo[0];
    const NvDsInferLayerInfo &landmLayer = outputLayersInfo[1];
    const NvDsInferLayerInfo &confLayer  = outputLayersInfo[2];

    const float* locData   = reinterpret_cast<const float*>(locLayer.buffer);
    const float* landmData = reinterpret_cast<const float*>(landmLayer.buffer);
    const float* confData  = reinterpret_cast<const float*>(confLayer.buffer);
    std::cout << "locData: " << locData << std::endl;
    // Determinar numBboxes (ej: 16800)
    size_t numBboxes = (locLayer.inferDims.numDims > 0) ? locLayer.inferDims.d[0] : 0;
    if (numBboxes == 0) {
        std::cerr << "ERROR: locLayer.inferDims.d[0] == 0." << std::endl;
        return false;
    }

    float confThreshold = 0.5; 
    float nmsThreshold  = 0.5; 
    batchSize = 1; // Forzamos a 1 para simplificar el código
    // Procesar cada imagen del batch
    for (int b = 0; b < batchSize; ++b) {
        // Calcular punteros para cada batch si la red se ejecuta con batch>1
        const float* locPtr   = locData   + b * numBboxes * 4;
        const float* landmPtr = landmData + b * numBboxes * 10;
        const float* confPtr  = confData  + b * numBboxes * 2;

        // Decodificar detecciones
        auto dets = decodeRetinaFace(locPtr, landmPtr, confPtr, inputW, inputH, confThreshold);

        // (Opcional) Aplicar NMS
        dets = applyNMS(dets, nmsThreshold);

        // Llenar la lista final de objetos
        for (auto &det : dets) {
            float score = det.confidence;

            std::cout << "Detection: " << score << " [" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]" << std::endl;
            
            if (score < confThreshold) continue;

            float x1 = det.x1;
            float y1 = det.y1;
            float x2 = det.x2;
            float y2 = det.y2;

            // Descartar bounding boxes degeneradas
            if ((x2 - x1) < 1.0f || (y2 - y1) < 1.0f) {
                continue;
            }

            // Agregar detección en formato DeepStream
            NvDsInferObjectDetectionInfo obj;
            obj.classId = 0;  // Asumiendo clase "rostro" = 0
            obj.detectionConfidence = score;
            obj.left   = x1;
            obj.top    = y1;
            obj.width  = x2 - x1;
            obj.height = y2 - y1;

            objectList.push_back(obj);
        }
    }

    return true;
}
