(ns dl4j.examples.nlp.glove.raw-text
  (:require [dl4j.examples.util :as u])
  (:import (org.deeplearning4j.models.glove Glove$Builder)))

;;;; There appears to be a problem with the GloVe implementation. Training fails frequently with a NaN error as described
;;;; here: https://github.com/deeplearning4j/dl4j-examples/issues/227. Decreasing the learning rate from 0.1 seems
;;;; to significantly decrease the frequency of this on my machine.

(defn vectors [iterator]
  (doto
    (-> (Glove$Builder.)
        (.iterate iterator)
        (.tokenizerFactory (u/basic-tokenizer))
        (.alpha 0.75)
        (.learningRate 0.099)
        (.epochs 25)
        (.xMax 100)
        (.batchSize 1000)
        (.shuffle true)
        (.symmetric true)
        (.build))
    (.fit)))

(comment

  (def v (vectors (u/text-file-iterator "raw_sentences.txt")))

  (.similarity v "day" "night")
  ; => 0.8967148661613464

  (.wordsNearest v "day" 10)
  ; => ["every" "may" "night" "year" "week" "game" "play" "time" "team" "work"]

  )
