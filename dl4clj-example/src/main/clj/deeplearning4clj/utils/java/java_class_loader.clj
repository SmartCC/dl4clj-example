(ns deeplearning4clj.utils.java.java-class-loader
  (:require [clojure.reflect :as cr]))

(defn class-exists?
  "Check if class c exists on the classpath"
  [c]
  (cr/resolve-class (.getContextClassLoader (Thread/currentThread)) c))

(defn nested-class
  "调用一个类的内部类，返回结果为内部类。如果类不存在则抛出ClassNotFoundException的异常"
  [parent-class nested-class-name]
  (let [class-name (-> (cr/typename parent-class) (str "$" nested-class-name) symbol)]
    (if (class-exists? class-name)
      (resolve class-name)
      (throw (ClassNotFoundException. (str class-name))))))

(defn class-static-var
  "执行类中的静态变量"
  [class-name var-name]
  (-> (str class-name "/" var-name) read-string eval))
