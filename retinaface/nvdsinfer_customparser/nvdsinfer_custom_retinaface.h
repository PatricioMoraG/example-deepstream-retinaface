/**
 * @file nvdsinfer_retinaface_parser.h
 * @brief Declaraciones del parser custom para RetinaFace en DeepStream.
 */

#ifndef NVDSINFER_RETINAFACE_PARSER_H
#define NVDSINFER_RETINAFACE_PARSER_H

#include <vector>
#include "nvdsinfer_custom_impl.h"

// Forward declaration opcional (si no incluyes .h de DeepStream completo)
// struct NvDsInferLayerInfo;
// struct NvDsInferNetworkInfo;
// struct NvDsInferParseDetectionParams;
// struct NvDsInferObjectDetectionInfo;
// struct NvDsInferAttribute;

/**
 * @brief Función principal del parser RetinaFace, exportada en C.
 *
 * @param outputLayersInfo        Salidas de la red (loc, landms, conf).
 * @param networkInfo             Info de la red (ancho, alto, canales).
 * @param detectionParams         Parámetros de umbrales, etc.
 * @param objectList              Vector de detecciones (BBs).
 * @param attrList                Vector de atributos (no se usa mucho en BB detection).
 * @param customData              Datos extra (ej: puntero a priors).
 * @param batchSize               Tamaño de batch (DeepStream).
 *
 * @return true si se parseó correctamente, false si hubo error.
 */
extern "C"
bool NvDsInferParseCustomRetinaFace(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList,
    std::vector<NvDsInferAttribute> &attrList,
    void* customData,
    int batchSize);

#endif // NVDSINFER_RETINAFACE_PARSER_H
