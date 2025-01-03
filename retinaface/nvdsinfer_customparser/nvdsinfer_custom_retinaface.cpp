/*******************************************************************************
 * Copyright (c) ...
 * All rights reserved.
 *
 * Custom parser para RetinaFace en DeepStream
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "nvdsinfer_custom_retinaface.h"
// #include <nvinfer.h>

// DeepStream 6.x / TensorRT 8.x
// Se asume que la librería se compila con -shared -fPIC y que se definirá
// la función de parseo con C linkage.

// -----------------------------------------------------
// Estructuras auxiliares
// -----------------------------------------------------
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} BBox;

// Variances típicas usadas en RetinaFace (puedes ajustarlas)
static const float VARIANCE[2] = {0.1f, 0.2f};

/**
 * @brief Decodifica las cajas (loc) a partir de priors, aplicando las variances.
 *        La representación asume:
 *          loc[:2] = (delta_cx, delta_cy)
 *          loc[2:] = (delta_w,  delta_h)
 *          prior[:2] = (cx, cy), prior[2:] = (w, h)
 */
static BBox
decodeBBox(const float loc[4], const float prior[4])
{
    // prior format: [cx, cy, w, h]
    // loc format:   [dx, dy, dw, dh]
    BBox box;

    float cx = prior[0] + loc[0] * VARIANCE[0] * prior[2];
    float cy = prior[1] + loc[1] * VARIANCE[0] * prior[3];
    float w  = prior[2] * std::exp(loc[2] * VARIANCE[1]);
    float h  = prior[3] * std::exp(loc[3] * VARIANCE[1]);

    // Convert (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
    box.x1 = cx - w * 0.5f;
    box.y1 = cy - h * 0.5f;
    box.x2 = cx + w * 0.5f;
    box.y2 = cy + h * 0.5f;

    return box;
}

// -----------------------------------------------------
// Aquí iría tu implementación para generar o leer los priors.
// -----------------------------------------------------
// En muchas implementaciones, se precalculan los priors en Python/C++
// y se cargan en un array. O se generan en tiempo de ejecución.
// Por simplicidad, asumimos que la persona ya tiene un array "priors"
// con shape [16800, 4], o algo similar, que pasamos al parser.
// -----------------------------------------------------

// -----------------------------------------------------
// Función principal de parseo
// -----------------------------------------------------

/**
 * @brief Implementación principal para parsear las salidas de RetinaFace.
 * 
 * @param outputLayersInfo Arreglo con la info de las salidas del engine:
 *        outputLayersInfo[0] => 16800x4  (localizaciones - loc)
 *        outputLayersInfo[1] => 16800x10 (landmarks - landms)
 *        outputLayersInfo[2] => 16800x2  (conf - fondo/rostro)
 * 
 * @param networkInfo  Dimensiones de la red (width, height, ...).
 * @param detectionThreshold Umbral de confianza para filtrar detecciones.
 * @param customData    Puntero a userData, si deseas pasar datos extra (ej. priors).
 * @param batchSize     Tamaño de batch (DeepStream puede procesar varios frames a la vez).
 *
 * @return Vector con las detecciones encontradas (NvDsInferObjectDetectionInfo).
 */
extern "C"
bool NvDsInferParseCustomRetinaFace(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList,
    std::vector<NvDsInferAttribute> &attrList,
    void* customData,
    int /*batchSize*/)
{
    // -----------------------------------------------------
    // 1) Validar que tengamos 3 salidas
    // -----------------------------------------------------
    if (outputLayersInfo.size() < 3) {
        std::cerr << "ERROR: Se esperan al menos 3 salidas [loc, landms, conf]." << std::endl;
        return false;
    }

    // -----------------------------------------------------
    // 2) Mapear punteros a cada salida
    // -----------------------------------------------------
    // Normalmente, sabrás cuál index corresponde a loc/conf/landms
    // según el orden configurado en tu engine o en el archivo de config.
    // Aquí asumimos:
    //   output0 => loc (16800x4)
    //   output1 => landms (16800x10)
    //   output2 => conf (16800x2)
    //
    // OJO: Revisa "layerName" si fuera necesario un mapeo por nombre.
    const NvDsInferLayerInfo &locLayer   = outputLayersInfo[0];
    const NvDsInferLayerInfo &landmLayer = outputLayersInfo[1];
    const NvDsInferLayerInfo &confLayer  = outputLayersInfo[2];

    // const float *locData   = reinterpret_cast<const float *>(locLayer.buffer);
    // const float *landmData = reinterpret_cast<const float *>(landmLayer.buffer);
    // const float *confData  = reinterpret_cast<const float *>(confLayer.buffer);
    const float* locData = static_cast<float*>(locLayer.buffer);
    const float* confData = static_cast<float*>(confLayer.buffer);

    // Numero total de "priors" o anchors
    //  Ej: 16800 => (40x40 + 20x20 + ...), depende de la config
    //  Asumimos que el tamaño se calcula de la dimensión: 16800 * 4 = total float en "loc"
    size_t numBboxes = locLayer.inferDims.d[0]; // usualmente 16800
    // locLayer.inferDims = (16800, 4)

    // -----------------------------------------------------
    // 3) Leer priors (o generarlos) - O se pasan en customData
    // -----------------------------------------------------
    // Se asume que 'customData' podría contener un float* a un array con priors.
    // p.ej: [16800, 4].
    if (!customData) {
        std::cerr << "ERROR: customData (priors) no esta definido." << std::endl;
        return false;
    }

    const float* priorData = reinterpret_cast<const float*>(customData);

    // -----------------------------------------------------
    // 4) Parsear y decodificar
    // -----------------------------------------------------
    objectList.clear();
    objectList.reserve(numBboxes);

    float detectionThreshold = detectionParams.perClassThreshold[0]; 
    // Nota: en DS, detectionParams.perClassThreshold se indexa por clase.
    // Si solo hay 1 clase (rostro), index = 0.

    for (size_t i = 0; i < numBboxes; ++i) {
        // conf para "rostro" se asume en canal 1 (confData[i*2 + 1])
        float score = confData[i * 2 + 1];
        if (score < detectionThreshold) {
            continue;
        }

        // Decodificar bbox
        const float* locPtr   = &locData[i * 4];
        const float* priorPtr = &priorData[i * 4];

        BBox box = decodeBBox(locPtr, priorPtr);

        // Convertir a coordenadas de la imagen final
        // (networkInfo.width, networkInfo.height)
        // Asumiendo que (box.x1, box.y1, box.x2, box.y2) están en [0,1],
        // depende de cómo generaste priors y normalizaste.
        // Ajusta si tus priors ya están en píxeles absolutos.

        float x1 = box.x1 * networkInfo.width;
        float y1 = box.y1 * networkInfo.height;
        float x2 = box.x2 * networkInfo.width;
        float y2 = box.y2 * networkInfo.height;

        // Clip a [0, ancho/alto] para evitar salir de la imagen
        x1 = std::max(0.0f, std::min(x1, (float)networkInfo.width - 1));
        y1 = std::max(0.0f, std::min(y1, (float)networkInfo.height - 1));
        x2 = std::max(0.0f, std::min(x2, (float)networkInfo.width - 1));
        y2 = std::max(0.0f, std::min(y2, (float)networkInfo.height - 1));

        if ((x2 - x1) < 1 || (y2 - y1) < 1) {
            // Descarta boxes degeneradas
            continue;
        }

        // Agregamos resultado al vector objectList
        NvDsInferObjectDetectionInfo objInfo;
        objInfo.classId = 0;  // Asumiendo "0" = rostro
        objInfo.detectionConfidence = score;
        objInfo.left = x1;
        objInfo.top  = y1;
        objInfo.width  = (x2 - x1);
        objInfo.height = (y2 - y1);

        objectList.push_back(objInfo);

        // -----------------------------------------------------
        // (Opcional) Si quieres landmarks, también puedes almacenarlos
        // en meta-datos extras, pero DeepStream, por defecto, se enfoca
        // en bounding boxes. Podrías usar "attrList" o user meta.
        // -----------------------------------------------------
        // Ejemplo rápido (no estándar):
        //   float lmk[10];
        //   const float* landmPtr = &landmData[i * 10];
        //   // decodificar con 'priorPtr' similar a decodeBBox
        //   // ...
    }

    return true;
}
