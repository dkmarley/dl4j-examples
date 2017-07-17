(defproject dl4j-examples "0.1.0-SNAPSHOT"
  :description "Examples from DL4J in Clojure"
  :url ""
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native-platform "0.4.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.4-rc3.8"]]
  :aot [#"dl4j.examples.nlp.iterators" #"dl4j.examples.nlp.tokenization"])
