(ns dl4j.examples.nlp.sequencevectors.sentences
  (:require [dl4j.examples.util :as u])
  (:import (org.deeplearning4j.models.embeddings WeightLookupTable)
           (org.deeplearning4j.models.embeddings.inmemory InMemoryLookupTable$Builder)
           (org.deeplearning4j.models.embeddings.learning.impl.elements SkipGram)
           (org.deeplearning4j.models.embeddings.loader VectorsConfiguration)
           (org.deeplearning4j.models.sequencevectors SequenceVectors$Builder)
           (org.deeplearning4j.models.sequencevectors.iterators AbstractSequenceIterator$Builder)
           (org.deeplearning4j.models.sequencevectors.transformers.impl SentenceTransformer$Builder)
           (org.deeplearning4j.models.word2vec VocabWord)
           (org.deeplearning4j.models.word2vec.wordstore VocabConstructor$Builder)
           (org.deeplearning4j.models.word2vec.wordstore.inmemory AbstractCache$Builder)
           (org.deeplearning4j.models.word2vec Word2Vec$Builder)))

(defn sentence-transformer [line-iterator]
  (-> (SentenceTransformer$Builder.)
      (.iterator line-iterator)
      (.tokenizerFactory (u/basic-tokenizer))
      (.build)))

(defn sequence-iterator [line-iterator]
  (-> (sentence-transformer line-iterator)
      (AbstractSequenceIterator$Builder.)
      (.build)))

(defn vectors [iterator]
  (let [vocab-cache      (-> (AbstractCache$Builder.) (.build))
        vocab-constructor (doto
                           (-> (VocabConstructor$Builder.)
                               (.addSource (sequence-iterator iterator) 5)
                               (.setTargetVocabCache vocab-cache)
                               (.build))
                           (.buildJointVocabulary false true))
        lookup-table     (doto
                           (-> (InMemoryLookupTable$Builder.)
                               (.lr 0.025)
                               (.vectorLength 150)
                               (.useAdaGrad false)
                               (.cache vocab-cache)
                               (.build))
                           (.resetWeights true))]
    (doto
      (-> (VectorsConfiguration.)
          (SequenceVectors$Builder.)
          (.minWordFrequency 5)
          (.lookupTable lookup-table)
          (.iterate (sequence-iterator iterator))
          (.vocabCache vocab-cache)
          (.batchSize 250)
          (.iterations 1)
          (.epochs 1)
          (.resetModel false)
          (.trainElementsRepresentation true)
          (.trainSequencesRepresentation false)
          (.elementsLearningAlgorithm (SkipGram.))
          (.build))
      (.fit))))

(comment

  (def v (vectors (u/text-file-iterator "raw_sentences.txt")))

  (.similarity v "state" "country")
  ; => 0.7050086855888367

  )
