(ns dl4j.examples.util
  (:require [clojure.java.io :as io])
  (:import (java.io FileNotFoundException)))

(defn slurp-from-classpath
  "Slurps a file from the classpath."
  [path]
  (or (some-> path
              io/resource
              slurp)
      (throw (FileNotFoundException. path))))

