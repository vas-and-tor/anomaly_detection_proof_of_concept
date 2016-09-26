#!/usr/bin/env python2

import pandas
import sys
import math

from context_ose.cad_ose import ContextualAnomalyDetectorOSE


if (len(sys.argv) < 2):
    print("Usage: " + sys.argv[0] + " [CSV_FILE]")
    exit(-1)

filePath = sys.argv[1]

dataSet = pandas.io.parsers.read_csv(filePath, header=0, parse_dates=[0])
inputMin = dataSet["value"].min()
inputMax = dataSet["value"].max()
probationaryPercent = 0.15
def getProbationPeriod(probationPercent, fileLength):
  """Return the probationary period index."""
  return min(
    math.floor(probationPercent * fileLength),
    probationPercent * 5000)
probationaryPeriod = getProbationPeriod(probationaryPercent, dataSet.shape[0])

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


cadose = ContextualAnomalyDetectorOSE (
  minValue = inputMin,
  maxValue = inputMax,
  restPeriod = probationaryPeriod / 5.0,
)


def handleRecord(inputData):
  anomalyScore = cadose.getAnomalyScore(inputData)
  return (anomalyScore,)

run()
