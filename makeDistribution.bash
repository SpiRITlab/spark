#!/bin/bash

./dev/make-distribution.sh --name SparkFHE --tgz -Phadoop-3.2.0 -Pmesos -Pyarn -Pkubernetes -Phive -Phive-thriftserver
