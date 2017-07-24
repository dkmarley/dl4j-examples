(ns dl4j.examples.nlp.word2vec.raw-text
  (:require [clojure.java.io :as io]
            [dl4j.examples.util :as u])
  (:import (org.deeplearning4j.models.word2vec Word2Vec$Builder)
           (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)))

(defn vectors [iterator]
  (doto
    (-> (Word2Vec$Builder.)
        (.minWordFrequency 5)
        (.iterations 1)
        (.layerSize 100)
        (.seed 42)
        (.windowSize 5)
        (.iterate iterator)
        (.tokenizerFactory (u/basic-tokenizer))
        (.build))
    (.fit)))

(comment

  ;; N.B. There are only 241 words in this vocabulary. See the classifier example for more impressive examples.
  (def v (vectors (u/text-file-iterator "raw_sentences.txt")))

  (.wordsNearest v "president" 10)
  ; => ["center" "state" "west" "government" "law" "federal" "police" "former" "director" "company"]

  (.wordsNearest v "country" 10)
  ; => ["house" "company" "center" "state" "group" "federal" "business" "national" "former" "program"]

  (.wordsNearest v "team" 10)
  ; => ["company" "program" "country" "national" "game" "season" "family" "war" "group" "market"]

  (.wordsNearest v "day" 10)
  ; => ["night" "week" "year" "game" "season" "time" "-" "office" "group" "show"]

  )
