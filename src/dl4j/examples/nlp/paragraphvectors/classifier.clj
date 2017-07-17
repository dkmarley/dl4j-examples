(ns maia.deeplearning.classifier
  (:require [clojure.java.io :as io]
            [dl4j.examples.util :as u]
            [dl4j.examples.vector :as v])
  (:import (org.deeplearning4j.models.paragraphvectors ParagraphVectors$Builder)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)
           (dl4j.examples.nlp.iterators ItemLabelAwareIterator)
           (dl4j.examples.nlp.tokenization TokenizerFactory)))

(defn training-items [path]
  (->> path
       (io/resource)
       (io/as-file)
       (.listFiles)
       (remove #(.isHidden %))
       (mapcat (fn [dir]
                 (let [class (.getName dir)]
                   (map
                     (fn [file]
                       {:class class :features (u/slurp-from-classpath (format "%s/%s/%s" path class (.getName file)))})
                     (.listFiles dir)))))
       (into [])))

;; Maybe use our own tokenizer
;; (def tokenizer (TokenizerFactory.))

(def tokenizer (doto (DefaultTokenizerFactory.) (.setTokenPreProcessor (CommonPreprocessor.))))

(defn vectors [iterator]
  (let [vectors (-> (ParagraphVectors$Builder.)
                    (.learningRate 0.025)
                    (.minLearningRate 0.001)
                    (.batchSize 1000)
                    (.epochs 20)
                    (.iterate iterator)
                    (.trainWordVectors true)
                    (.tokenizerFactory tokenizer)
                    (.build))]
    (.fit vectors)
    vectors))

(defn tokenize [tokenizer text]
  (-> (.create tokenizer text) (.getTokens)))


(defn classify [vectors iterator tokenizer text]
  (let [vector-table (.getLookupTable vectors)
        labels       (-> iterator (.getLabelsSource) (.getLabels))
        centroid     (->> text
                          (tokenize tokenizer)
                          (v/vocab-tokens (.getVocab vectors))
                          (v/tokens->vector vector-table)
                          (v/centroid))]
    (v/similar-vectors vector-table labels centroid)))

(defn classifier [items]
  (let [iterator (ItemLabelAwareIterator. items)
        vectors  (vectors iterator)]
    (fn [text] (classify vectors iterator tokenizer text))))

(comment

  ;; Instead of creating the iterator, training the model with it and calling 'classify', we could create a
  ;; classifier with 'classifier' above but we want access to the word vectors for experimentation.
  (def i (ItemLabelAwareIterator. (training-items "paravec/labeled")))

  (def v (vectors i))

  (classify v i tokenizer (u/slurp-from-classpath "paravec/unlabeled/finance/f01.txt"))
  ;; => (["finance" 0.42245012521743774] ["science" -0.00961653608828783] ["health" -0.028964674100279808])

  (classify v i tokenizer (u/slurp-from-classpath "paravec/unlabeled/health/f01.txt"))
  ;; => (["health" 0.5211763381958008] ["science" 0.001582252443768084] ["finance" -0.3309013843536377])

  (.wordsNearest v "bank" 10)
  ;; => #{"suisse" "jpmorgan" "sachs" "fact" "blocked" "citi" "takeover" "goldman" "deutsche" "described"}

  (.wordsNearest v "oil" 10)
  ;; => #{"government" "like" "contract" "gas" "standardized" "gasoline" "prices" "heating" "futures" "crude"}

  (.wordsNearest v "learning" 10)
  ;; => #{"algorithms" "discover" "semi-supervised" "learn" "unsupervised" "find" "learns" "multiple" "low-dimensional" "representation"}}

  )
