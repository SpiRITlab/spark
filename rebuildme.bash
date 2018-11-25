#!/usr/bin/env bash

build/mvn -DskipTests package
build/mvn -DskipTests install
