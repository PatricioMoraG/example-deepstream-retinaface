/*******************************************************************************
 * nvdsinfer_custom_retinaface.h
 *
 * Custom parser para RetinaFace en DeepStream.
 ******************************************************************************/

#ifndef NVDSINFER_CUSTOM_RETINAFACE_H
#define NVDSINFER_CUSTOM_RETINAFACE_H

#include <vector>

// Incluir los encabezados de DeepStream necesarios
// Asegúrate de que el SDK de DeepStream esté correctamente instalado y configurado.
#include "nvdsinfer.h"

/**
 * @brief Estructura auxiliar para stride y anchor base.
 */
struct StrideAnchor {
    int stride;      /**< Tamaño del stride (ej. 8, 16, 32) */
    int baseAnchor;  /**< Tamaño base del anchor (ej. 16, 64, 256) */
};

/**
 * @brief Estructura que representa una detección de RetinaFace.
 */
struct RetinaFaceDetection {
    float x1;           /**< Coordenada x del borde izquierdo de la caja */
    float y1;           /**< Coordenada y del borde superior de la caja */
    float x2;           /**< Coordenada x del borde derecho de la caja */
    float y2;           /**< Coordenada y del borde inferior de la caja */
    float confidence;   /**< Puntaje de confianza de la detección */
    float landmarks[10];/**< Coordenadas de los 5 puntos de referencia (x, y) */
};

/**
 * @brief Decodifica las salidas de RetinaFace para obtener las detecciones.
 *
 * @param locData Puntero a los datos de localización.
 * @param landmData Puntero a los datos de landmarks.
 * @param confData Puntero a los datos de confianza.
 * @param inputWidth Ancho de entrada de la imagen.
 * @param inputHeight Alto de entrada de la imagen.
 * @param confThreshold Umbral de confianza para filtrar detecciones.
 *
 * @return Un vector de detecciones de RetinaFace que cumplen con el umbral de confianza.
 */
std::vector<RetinaFaceDetection> decodeRetinaFace(
    const float* locData,
    const float* landmData,
    const float* confData,
    int inputWidth,      /**< Ancho de entrada de la red (INPUT_W) */
    int inputHeight,     /**< Alto de entrada de la red (INPUT_H) */
    float confThreshold  /**< Umbral de confianza (ej. 0.5 o 0.6) */
);

/**
 * @brief Función principal para parsear las salidas de RetinaFace en DeepStream.
 *
 * Esta función es llamada por DeepStream para interpretar las salidas de la red y
 * extraer las detecciones de objetos (rostros) en el formato requerido por DeepStream.
 *
 * @param outputLayersInfo Vector que contiene información sobre las capas de salida de la red.
 * @param networkInfo Información sobre la red, como dimensiones de entrada.
 * @param detectionParams Parámetros adicionales para la detección.
 * @param objectList Vector donde se almacenarán las detecciones de objetos.
 * @param attrList Vector donde se almacenarán los atributos adicionales (si los hay).
 * @param customData Puntero a datos personalizados (no utilizado en este caso).
 * @param batchSize Tamaño del batch (número de imágenes procesadas simultáneamente).
 *
 * @return `true` si el parsing fue exitoso, `false` de lo contrario.
 */
extern "C" bool NvDsInferParseCustomRetinaFace(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList,
    std::vector<NvDsInferAttribute> &attrList,
    void* customData,
    int batchSize
);

#endif // NVDSINFER_CUSTOM_RETINAFACE_H
