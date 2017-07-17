(ns dl4j.examples.nlp.tokenization.TokenizerFactory
  (:gen-class :implements [org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory]
              :constructors {[] []}
              :methods [])
  (:import (dl4j.examples.nlp.tokenization Tokenizer)))

(defn- -create [this text]
  (Tokenizer. text))