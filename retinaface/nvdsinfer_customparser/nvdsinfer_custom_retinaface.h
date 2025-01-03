/******************************************************************************
 * nvdsinfer_custom_retinaface.h
 * 
 * Custom parser para RetinaFace en DeepStream
 ******************************************************************************/

#ifndef NVDSINFER_CUSTOM_RETINAFACE_H
#define NVDSINFER_CUSTOM_RETINAFACE_H

#include <vector>

// Asegúrate de incluir el header adecuado que declara `NvDsInferParseDetectionParams`
// Normalmente se encuentra en "nvdsinfer_custom_impl.h" o "nvdsinfer.h", dependiendo
// de tu versión de DeepStream. Ajusta según tu instalación.
#include "nvdsinfer_custom_impl.h" 

//------------------------------------------------------------------------------
// Estructuras auxiliares y declaraciones de funciones
//------------------------------------------------------------------------------

/**
 * @brief Estructura auxiliar para stride y anchor base.
 */
struct StrideAnchor {
    int stride;      /**< Tamaño del stride (ej. 8, 16, 32) */
    int baseAnchor;  /**< Tamaño base del anchor (ej. 16, 64, 256) */
};

/**
 * @brief Estructura que representa una detección de RetinaFace, con bbox y landmarks.
 */
struct RetinaFaceDetection {
    float x1;         /**< Coordenada x del borde izquierdo de la caja */
    float y1;         /**< Coordenada y del borde superior de la caja */
    float x2;         /**< Coordenada x del borde derecho de la caja */
    float y2;         /**< Coordenada y del borde inferior de la caja */
    float confidence; /**< Puntaje de confianza de la detección */
    float landmarks[10]; /**< Coordenadas de los 5 puntos de referencia (x,y) */
};

/**
 * @brief Decodifica las salidas de la red RetinaFace para generar detecciones.
 *
 * @param locData      Puntero a la data de localización (boxes).
 * @param landmData    Puntero a la data de landmarks.
 * @param confData     Puntero a la data de confianza (cls).
 * @param inputWidth   Ancho de la imagen de entrada.
 * @param inputHeight  Alto de la imagen de entrada.
 * @param confThreshold Umbral mínimo de confianza para filtrar detecciones.
 *
 * @return std::vector<RetinaFaceDetection> con las detecciones generadas.
 */
std::vector<RetinaFaceDetection> decodeRetinaFace(
    const float* locData,
    const float* landmData,
    const float* confData,
    int inputWidth,
    int inputHeight,
    float confThreshold
);

/**
 * @brief Parser principal que DeepStream llama para convertir las salidas de la red en
 *        NvDsInferObjectDetectionInfo y NvDsInferAttribute.
 *
 * @param outputLayersInfo Información de las capas de salida.
 * @param networkInfo      Información de la red (dimensiones de entrada).
 * @param detectionParams  Parámetros de detección de DeepStream (threshold, NMS, etc.).
 * @param objectList       Vector donde se almacenan las detecciones.
 * @param attrList         Vector donde se almacenan atributos adicionales (si se usan).
 * @param customData       Puntero a datos personalizados (opcional).
 * @param batchSize        Tamaño del batch.
 *
 * @return `true` si tuvo éxito, `false` en caso de error.
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
