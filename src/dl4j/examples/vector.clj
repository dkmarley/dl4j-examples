(ns dl4j.examples.vector
  (:require [clojure.java.io :as io])
  (:import (java.io FileNotFoundException)
           (org.nd4j.linalg.ops.transforms Transforms)
           (org.nd4j.linalg.factory Nd4j)))

(defn slurp-from-classpath
  "Slurps a file from the classpath."
  [path]
  (or (some-> path
              io/resource
              slurp)
      (throw (FileNotFoundException. path))))

(defn vocab-tokens [vocab tokens]
  (filter #(.containsWord vocab %) tokens))

(defn tokens->vector [vector-table tokens]
  (let [vector (Nd4j/create (count tokens) (.layerSize vector-table))
        tokens (map-indexed (fn [i token] [i token]) tokens)]
    (reduce (fn [v [index token]] (.putRow v index (.vector vector-table token)) v) vector tokens)))

(defn centroid [vector]
  (.mean vector (into-array Integer/TYPE [0])))

(defn norm [vector]
  (.norm2 vector (into-array Integer/TYPE [0])))

(defn similar-vectors [vector-table vector-labels vector]
  (let [compute-similarity (fn [label]
                             (let [v-label (.vector vector-table label)]
                               [label (Transforms/cosineSim vector v-label)]))]
    (->> (map compute-similarity vector-labels)
         (sort-by second >))))
