(ns dl4j.examples.util
  (:require [clojure.java.io :as io])
  (:import (java.io FileNotFoundException)
           (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)))

(defn slurp-from-classpath
  "Slurps a file from the classpath."
  [path]
  (or (some-> path
              io/resource
              slurp)
      (throw (FileNotFoundException. path))))

(defn file-from-classpath [path]
  (or (some-> (io/resource path)
              (io/as-file))
      (throw (FileNotFoundException. path))))

(defn text-file-iterator [path]
  (BasicLineIterator. (file-from-classpath path)))

(defn basic-tokenizer []
  (doto (DefaultTokenizerFactory.) (.setTokenPreProcessor (CommonPreprocessor.))))

