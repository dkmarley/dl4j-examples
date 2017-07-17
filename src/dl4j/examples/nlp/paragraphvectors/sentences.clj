(ns dl4j.examples.nlp.paragraphvectors.sentences
  (:require [clojure.java.io :as io]
            [dl4j.examples.util :as u])
  (:import (org.deeplearning4j.models.paragraphvectors ParagraphVectors$Builder)
           (org.deeplearning4j.models.word2vec.wordstore.inmemory InMemoryLookupCache)
           (org.deeplearning4j.text.documentiterator LabelsSource)
           (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)))

(defn training-file [path]
  (-> (io/resource path) (io/as-file)))

(defn iterator [path]
  (BasicLineIterator. (training-file path)))

(defn vectors [iterator]
  (let [cache     (InMemoryLookupCache.)
        labels    (LabelsSource. "DOC_")
        tokenizer (doto (DefaultTokenizerFactory.) (.setTokenPreProcessor (CommonPreprocessor.)))
        vectors   (-> (ParagraphVectors$Builder.)
                      (.iterations 5)
                      (.epochs 1)
                      (.layerSize 100)
                      (.learningRate 0.025)
                      (.minWordFrequency 1)
                      (.labelsSource labels)
                      (.windowSize 5)
                      (.iterate iterator)
                      (.trainWordVectors false)
                      (.vocabCache cache)
                      (.tokenizerFactory tokenizer)
                      (.sampling 0.0)
                      (.build))]
    (.fit vectors)
    vectors))

(comment

  (def v (vectors (iterator "raw_sentences.txt")))

  (println "9836/12493 ('This is my house .'/'This is my world .') similarity:" (.similarity v "DOC_9835" "DOC_12492"))
  ;; similarity: 0.7573885917663574

  (println "6348/3721 ('This is my case .'/'This is my way .') similarity:" (.similarity v "DOC_6347" "DOC_3720"))
  ;; similarity: 0.917813777923584

  (println "3721/9853 ('This is my way .'/'We now have one .') similarity:" (.similarity v "DOC_3720" "DOC_9852"))
  ;; similarity: 0.5390669703483582

  )
