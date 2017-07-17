(ns dl4j.examples.nlp.tokenization.Tokenizer
  (:gen-class :implements [org.deeplearning4j.text.tokenization.tokenizer.Tokenizer]
              :constructors {[java.lang.String] []}
              :init init
              :state state
              :methods [])
  (:require [clojure.string :as s]))

(defn- -init
  ([text]
   [[] (ref {:text text})]))

(defn- -getTokens
  "More or less the same as DefaultTokenizer with CommonPreProcessor. Provided here as a convenience for trying
  different tokenizations. E.g. maybe add stemming?"
  [this]
  (let [{:keys [text]} @(.state this)]
    (-> text
        (s/replace #"[\d\.:,\"\'\(\)\[\]|/?!;]+" "")
        (s/lower-case)
        (s/split #"\s"))))