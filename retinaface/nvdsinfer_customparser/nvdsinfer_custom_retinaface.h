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
 * Genera anchors (priors) para RetinaFace, equivalentes al Python PriorBox.
 *
 * @param min_sizes  Colección de min_sizes por nivel. Ej.: {{16,32}, {64,128}, {256,512}}
 * @param steps      Distancias en píxeles de cada nivel. Ej.: {8,16,32}
 * @param input_height Altura de la imagen de entrada (ej.: 640 u 840)
 * @param input_width  Ancho  de la imagen de entrada
 * @param clip       Indica si se deben recortar los valores a [0,1]
 *
 * @return Un std::vector<float> con todas las anchors concatenadas en formato
 *         [cx, cy, w, h,  cx, cy, w, h,  ...].
 */
std::vector<float> generate_retinaface_anchors(
    const std::vector<std::vector<int>>& min_sizes,
    const std::vector<int>& steps,
    int input_height,
    int input_width,
    bool clip
)

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
