(ns dl4j.examples.nlp.iterators.ItemLabelAwareIterator
  (:import (org.deeplearning4j.text.documentiterator LabelledDocument LabelsSource))
  (:gen-class :implements [org.deeplearning4j.text.documentiterator.LabelAwareIterator]
              :constructors {[clojure.lang.Indexed] []}
              :init init
              :state state
              :methods []))

;;;; An iterator that takes a sequence of {:class label :features the-features} and provides LabelledDocuments

(defn- -init
  ([items]
   [[] (ref {:items items
             :labels (distinct (map :class items))
             :index 0})]))

(defn- -hasNext [this]
  (let [{:keys [items index]} @(.state this)]
    (< index (count items))))

(defn- -hasNextDocument [this]
  (-hasNext this))

(defn- -next [this]
  (let [{:keys [items index]}    @(.state this)
        {:keys [class features]} (get items index)]
    (when (not (-hasNext this))
      (throw (IndexOutOfBoundsException. "There is no next document")))
    (dosync (alter (.state this) assoc :index (inc index)))
    (doto (LabelledDocument.)
      (.setContent features)
      (.setLabel class))))

(defn- -nextDocument [this]
  (-next this))

(defn- -reset [this]
  (dosync (alter (.state this) assoc :index 0)))

(defn- -shutdown [this])

(defn -getLabelsSource [this]
  (let [{:keys [labels]} @(.state this)]
    (LabelsSource. labels)))

