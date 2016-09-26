#!/usr/bin/env python2

import pandas
import sys
import math

from skyline.algorithms import (median_absolute_deviation,
                                              first_hour_average,
                                              stddev_from_average,
                                              stddev_from_moving_average,
                                              mean_subtraction_cumulation,
                                              least_squares,
                                              histogram_bins)


if (len(sys.argv) < 2):
    print("Usage: " + sys.argv[0] + " [CSV_FILE]")
    exit(-1)

filePath = sys.argv[1]

dataSet = pandas.io.parsers.read_csv(filePath, header=0, parse_dates=[0])

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


timeseries = []
recordCount = 0
algorithms =   [median_absolute_deviation,
                     first_hour_average,
                     stddev_from_average,
                     stddev_from_moving_average,
                     mean_subtraction_cumulation,
                     least_squares,
                     histogram_bins]


def handleRecord(inputData):
  """
  Returns a list [anomalyScore].
  """

  score = 0.0
  inputRow = [inputData["timestamp"], inputData["value"]]
  timeseries.append(inputRow)
  for algo in algorithms:
    score += algo(timeseries)

  averageScore = score / (len(algorithms) + 1)
  return [averageScore]

run()
