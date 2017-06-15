(ns deeplearning4clj.utils.java.keyword-2-java-test
  (:require [clojure.test :refer :all]
            [deeplearning4clj.utils.java.keyword-2-java :refer :all])
  (:import [java.util HashMap]))

(deftest keyword-2-java-test
  (is (= (init-object-&-run doto-keyword-2-java HashMap :put [:a 1] :put [:b 2]) {:a 1 :b 2})
      "关键词转java函数测试失败，测试内容为HashMap"))

