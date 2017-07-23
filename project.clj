(defproject dl4j-examples "0.1.1-SNAPSHOT"
  :description "Examples from DL4J in Clojure"
  :url "https://github.com/dkmarley/dl4j-examples"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.slf4j/slf4j-log4j12 "1.7.25"]
                 [org.nd4j/nd4j-native-platform "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.8.0"]]
  :aot [#"dl4j.examples.nlp.iterators" #"dl4j.examples.nlp.tokenization"])
