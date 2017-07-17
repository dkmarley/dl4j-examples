(ns dl4j.examples.nlp.word2vec.raw-text
  (:require [clojure.java.io :as io]
            [dl4j.examples.util :as u]
            [dl4j.examples.vector :as v])
  (:import (org.deeplearning4j.models.word2vec Word2Vec$Builder)
           (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)))

(defn training-file [path]
  (-> (io/resource path) (io/as-file)))

(defn iterator [path]
  (BasicLineIterator. (training-file path)))

(defn vectors [iterator]
  (let [tokenizer (doto (DefaultTokenizerFactory.) (.setTokenPreProcessor (CommonPreprocessor.)))
        vectors   (-> (Word2Vec$Builder.)
                      (.minWordFrequency 5)
                      (.iterations 1)
                      (.layerSize 100)
                      (.seed 42)
                      (.windowSize 5)
                      (.iterate iterator)
                      (.tokenizerFactory tokenizer)
                      (.build))]
    (.fit vectors)
    vectors))

(comment

  ;; N.B. There are only 241 words in this vocabulary. See the classifier example for more impressive examples.
  (def v (vectors (iterator "raw_sentences.txt")))

  (.wordsNearest v "president" 10)
  ;; => #{"law" "police" "government" "united" "director" "center" "west" "federal" "state" "former"}

  (.wordsNearest v "country" 10)
  ;; #{"law" "world" "business" "center" "federal" "national" "company" "state" "house" "group"}

  (.wordsNearest v "team" 10)
  ;; => #{"market" "country" "game" "business" "national" "war" "company" "program" "family" "group"}

  )