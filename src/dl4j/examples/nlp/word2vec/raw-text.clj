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
  ;; => ["center" "state" "west" "government" "law" "federal" "police" "former" "director" "company"]

  (.wordsNearest v "country" 10)
  ;; => ["house" "company" "center" "state" "group" "federal" "business" "national" "former" "program"]

  (.wordsNearest v "team" 10)
  ;; => ["company" "program" "country" "national" "game" "season" "family" "war" "group" "market"]

  )
