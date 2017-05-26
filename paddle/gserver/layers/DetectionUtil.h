/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <float.h>
#include <algorithm>
#include <vector>
#include "paddle/math/Matrix.h"

using std::vector;
using std::pair;
using std::map;

namespace paddle {

template <typename T>
struct BBoxBase {
  BBoxBase(T xMin, T yMin, T xMax, T yMax)
      : xMin(xMin), yMin(yMin), xMax(xMax), yMax(yMax) {}

  BBoxBase() {}

  T getWidth() const { return xMax - xMin; }

  T getHeight() const { return yMax - yMin; }

  T getCenterX() const { return (xMin + xMax) / 2; }

  T getCenterY() const { return (yMin + yMax) / 2; }

  T getSize() const { return getWidth() * getHeight(); }

  T xMin;
  T yMin;
  T xMax;
  T yMax;
};

struct NormalizedBBox : BBoxBase<real> {
  NormalizedBBox() : BBoxBase<real>() {}
};

enum PermMode { NCHWTONHWC, NHWCTONCHW };

/**
 * @brief First permute input maxtrix then append to output matrix
 */
size_t appendWithPermute(const MatrixPtr inMatrix,
                         size_t height,
                         size_t width,
                         size_t outTotalSize,
                         size_t outOffset,
                         size_t batchSize,
                         MatrixPtr outMatrix,
                         PermMode permMode,
                         bool useGpu);

/**
 * @brief First permute input maxtrix then decompose to output
 */
size_t decomposeWithPermute(const MatrixPtr inMatrix,
                            size_t height,
                            size_t width,
                            size_t totalSize,
                            size_t offset,
                            size_t batchSize,
                            MatrixPtr outMatrix,
                            PermMode permMode,
                            bool useGpu);

/**
 * @brief Compute jaccard overlap between two bboxes.
 * @param bbox1 The first bbox
 * @param bbox2 The second bbox
 */
real jaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

/**
 * @brief Compute offset parameters between prior bbox and groundtruth bbox
 * and variances of prior bbox are considered
 * @param priorBBox Input prior bbox
 * @param priorBBoxVar Variance parameters of prior bbox
 * @param gtBBox Groundtruth bbox
 */
vector<real> encodeBBoxWithVar(const NormalizedBBox& priorBBox,
                               const vector<real> priorBBoxVar,
                               const NormalizedBBox& gtBBox);

/**
 * @brief Decode prior bbox with offset parameters
 * and variances of prior bbox are considered
 * @param priorBBox Prior bbox to be decoded
 * @param priorBBoxVar Variance parameters of prior bbox
 * @param locPredData Offset parameters
 */
NormalizedBBox decodeBBoxWithVar(const NormalizedBBox& priorBBox,
                                 const vector<real>& priorBBoxVar,
                                 const vector<real>& locPredData);

/**
 * @brief Extract bboxes from prior matrix, the layout is
 * xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var | ymax1Var ...
 * @param priorData Matrix of prior value
 * @param numBBoxes Number of bbox to be extracted
 * @param bboxVec Append to the vector
 */
void getBBoxFromPriorData(const real* priorData,
                          const size_t numBBoxes,
                          vector<NormalizedBBox>& bboxVec);

/**
 * @brief Extract variances from prior matrix, the layout is
 * xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var | ymax1Var ...
 * @param priorData Matrix of prior value
 * @param num Number to be extracted
 * @param varVec Append to the vector
 */
void getBBoxVarFromPriorData(const real* priorData,
                             const size_t num,
                             vector<vector<real>>& varVec);

/**
 * @brief Extract bboxes from label matrix, the layout is
 * class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ...
 * @param labelData Matrix of label value
 * @param numBBoxes Number to be extracted
 * @param bboxVec Append to the vector
 */
void getBBoxFromLabelData(const real* labelData,
                          const size_t numBBoxes,
                          vector<NormalizedBBox>& bboxVec);

/**
* @brief Match prior bbox to groundtruth bbox, the strategy is:
1. Find the most overlaped bbox pair (prior and groundtruth)
2. For rest of prior bboxes find the most overlaped groundtruth bbox
* @param priorBBoxes prior bbox
* @param gtBBoxes groundtruth bbox
* @param overlapThreshold Low boundary of overlap (judge whether matched)
* @param matchIndices For each prior bbox, groundtruth bbox index if matched
otherwise -1
* @param matchOverlaps For each prior bbox, overap with all groundtruth bboxes
*/
void matchBBox(const vector<NormalizedBBox>& priorBBoxes,
               const vector<NormalizedBBox>& gtBBoxes,
               real overlapThreshold,
               vector<int>* matchIndices,
               vector<real>* matchOverlaps);

/**
* @brief Generate positive bboxes and negative bboxes,
|positive bboxes|/|negative bboxes| is negPosRatio
* @param priorValue Prior value
* @param numPriorBBoxes Number of prior bbox
* @param gtValue Groundtruth value
* @param gtStartPosPtr Since groundtruth value stored as sequence type,
this parameter indicates start position of each record
* @param seqNum Number of sequence
* @param maxConfScore Classification score for prior bbox, used to mine
negative examples
* @param batchSize Image number
* @param overlapThreshold Low boundary of overap
* @param negOverlapThreshold Upper boundary of overap (judge negative example)
* @param negPosRatio Control number of negative bboxes
* @param matchIndicesVecPtr Save indices of matched prior bbox
* @param negIndicesVecPtr Save indices of negative prior bbox
*/
pair<size_t, size_t> generateMatchIndices(
    const MatrixPtr priorValue,
    const size_t numPriorBBoxes,
    const MatrixPtr gtValue,
    const int* gtStartPosPtr,
    const size_t seqNum,
    const vector<vector<real>>& maxConfScore,
    const size_t batchSize,
    const real overlapThreshold,
    const real negOverlapThreshold,
    const size_t negPosRatio,
    vector<vector<int>>* matchIndicesVecPtr,
    vector<vector<int>>* negIndicesVecPtr);

/**
 * @brief Get max confidence score for each prior bbox
 * @param confData Confidence scores, layout is
 * class1 score | class2 score | ... | classN score ...
 * @param batchSize Image number
 * @param numPriorBBoxes Prior bbox number
 * @param numClasses Classes number
 * @param backgroundId Background id
 * @param maxConfScoreVecPtr Ouput
 */
void getMaxConfidenceScores(const real* confData,
                            const size_t batchSize,
                            const size_t numPriorBBoxes,
                            const size_t numClasses,
                            const size_t backgroundId,
                            vector<vector<real>>* maxConfScoreVecPtr);

template <typename T>
bool sortScorePairDescend(const pair<real, T>& pair1,
                          const pair<real, T>& pair2);

}  // namespace paddle
