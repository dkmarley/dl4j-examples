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
  (->> (u/file-from-classpath path)
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

(def tokenizer (u/basic-tokenizer))

(defn vectors [iterator]
  (doto
    (-> (ParagraphVectors$Builder.)
        (.learningRate 0.025)
        (.minLearningRate 0.001)
        (.batchSize 1000)
        (.epochs 20)
        (.iterate iterator)
        (.trainWordVectors true)
        (.tokenizerFactory tokenizer)
        (.build))
    (.fit)))

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

(defn classifier [tokenizer items]
  (let [iterator (ItemLabelAwareIterator. items)
        vectors  (vectors iterator)]
    (fn [text] (classify vectors iterator tokenizer text))))

(comment

  ;; Instead of creating the iterator, training the model with it and calling 'classify', we could create a
  ;; classifier with 'classifier' above but we want access to the word vectors for experimentation.
  (def i (ItemLabelAwareIterator. (training-items "paravec/labeled")))

  (def v (vectors i))

  (classify v i tokenizer (u/slurp-from-classpath "paravec/unlabeled/finance/f01.txt"))
  ; => (["finance" 0.7510349750518799] ["science" -0.16607993841171265] ["health" -0.3732961118221283])

  (classify v i tokenizer (u/slurp-from-classpath "paravec/unlabeled/health/f01.txt"))
  ; => (["health" 0.5860732197761536] ["finance" -0.07874766737222672] ["science" -0.09755710512399673])

  (.wordsNearest v "bank" 10)
  ; => ["citi" "suisse" "goldman" "sachs" "jpmorgan" "merrill" "deutsche" "credit" "limited-purpose" "serve"]

  (.wordsNearest v "oil" 10)
  ; => ["heating" "prices" "gasoline" "gas" "crude" "palladium" "effect" "platinum" "transparent" "standardized"]

  (.wordsNearest v "learning" 10)
  ; => ["algorithms" "learn" "machine" "recognition" "computational" "aim" "multilinear" "signal" "tasks" "infeasible"]

  )
