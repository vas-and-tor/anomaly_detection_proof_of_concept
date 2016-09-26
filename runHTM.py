#!/usr/bin/env python2

import pandas
import sys
import math

from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.common_models.cluster_params import (
  getScalarMetricWithTimeOfDayAnomalyParams)
from nupic.frameworks.opf.modelfactory import ModelFactory

def _setupEncoderParams(encoderParams):
  # The encoder must expect the NAB-specific datafile headers
  encoderParams["timestamp_dayOfWeek"] = encoderParams.pop("c0_dayOfWeek")
  encoderParams["timestamp_timeOfDay"] = encoderParams.pop("c0_timeOfDay")
  encoderParams["timestamp_timeOfDay"]["fieldname"] = "timestamp"
  encoderParams["timestamp_timeOfDay"]["name"] = "timestamp"
  encoderParams["timestamp_weekend"] = encoderParams.pop("c0_weekend")
  encoderParams["value"] = encoderParams.pop("c1")
  encoderParams["value"]["fieldname"] = "value"
  encoderParams["value"]["name"] = "value"

def getProbationPeriod(probationPercent, fileLength):
  """Return the probationary period index."""
  return min(
    math.floor(probationPercent * fileLength),
    probationPercent * 5000)

if (len(sys.argv) < 2):
    print("Usage: " + sys.argv[0] + " [CSV_FILE]")
    exit(-1)

filePath = sys.argv[1]

dataSet = pandas.io.parsers.read_csv(filePath, header=0, parse_dates=[0])
inputMin = dataSet["value"].min()
inputMax = dataSet["value"].max()
rangePadding = abs(inputMax - inputMin) * 0.2
modelParams = getScalarMetricWithTimeOfDayAnomalyParams(
  metricData=[0],
  minVal=inputMin-rangePadding,
  maxVal=inputMax+rangePadding,
  minResolution=0.001,
  tmImplementation = "cpp"
)["modelConfig"]
_setupEncoderParams(modelParams["modelParams"]["sensorParams"]["encoders"])

model = ModelFactory.create(modelParams)
model.enableInference({"predictedField": "value"})

probationaryPercent = 0.15
probationaryPeriod = getProbationPeriod(probationaryPercent, dataSet.shape[0])
useLikelihood = True

numentaLearningPeriod = math.floor(probationaryPeriod / 2.0)
anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
  learningPeriod=numentaLearningPeriod,
  estimationSamples=probationaryPeriod-numentaLearningPeriod,
  reestimationPeriod=100
)

def run():
  """
  Main function that is called to collect anomaly scores for a given file.
  """

  import matplotlib.pyplot as plt

  anomalies_t = []
  anomalies_v = []
  anomalies_c = []

  all_t = []
  all_v = []

  rows = []
  for i, row in dataSet.iterrows():

    inputData = row.to_dict()

    detectorValues = handleRecord(inputData)

    if (detectorValues[0] > 0.65):
      anomalies_t.append(inputData["timestamp"])
      anomalies_v.append(inputData["value"])
      anomalies_c.append(detectorValues[0])

    all_t.append(inputData["timestamp"])
    all_v.append(inputData["value"])

    outputRow = list(row) + list(detectorValues)

    rows.append(outputRow)

    # Progress report
    if (i % 1000) == 0:
      print ".",
      sys.stdout.flush()

  fig, ax = plt.subplots()

  ax.plot(all_t, all_v)
  ax.plot(anomalies_t, anomalies_v, 'ro')

  plt.show()

  ans = pandas.DataFrame(rows)
  return ans


def handleRecord(inputData):
  """Returns a tuple (anomalyScore, rawScore).

  Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
  and "rawScore" corresponds to "anomaly_score". Sorry about that.
  """
  # Send it to Numenta detector and get back the results
  result = model.run(inputData)

  # Retrieve the anomaly score and write it to a file
  rawScore = result.inferences["anomalyScore"]

  if useLikelihood:
    # Compute log(anomaly likelihood)
    anomalyScore = anomalyLikelihood.anomalyProbability(
      inputData["value"], rawScore, inputData["timestamp"])
    logScore = anomalyLikelihood.computeLogLikelihood(anomalyScore)
    return (logScore, rawScore)

  return (rawScore, rawScore)

run()
