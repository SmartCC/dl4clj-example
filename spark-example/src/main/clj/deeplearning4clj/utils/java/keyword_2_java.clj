(ns deeplearning4clj.utils.java.keyword-2-java
  (:require [clojure.string :as s]))

(defn keyword-2-java-style
  "转换关键字风格为java风格，去掉每个-连接符并将其后的首字母大写，如to-string转为toString"
  [x]
  (let [[a & b] (s/split x #"-")]
    (apply str a (map s/capitalize b))))

(defmacro transfer-keyword-2-java-function
  "转换keyword为java函数，keyword可用写成clojure风格的形式，如:to-string，也可以写成java风格，如：toString，第二个参数为调用函数的java对象，第三个参数为java函数的参数可以是无参也可以是多个参数"
  [k j-object & forms]
  (let [f-forms (repeatedly (count forms) gensym)
        f `(memfn ~(-> (name k) keyword-2-java-style symbol) ~@f-forms)]
    `(~f ~j-object ~@forms)))

(defmacro keyword-2-java
  "处理函数的输入为多个参数的情况，在使用key-value的形式是，无法使一个key对应多个输入，故将输入组成一个序列，如：(def m (HashMap.)) (keyword-2-java m :put [\"a\" 2])。"
  ([j-object k] `(transfer-keyword-2-java-function ~k ~j-object))
  ([j-object k form]
   (cond
     (nil? form) `(transfer-keyword-2-java-function ~k ~j-object)
     (vector? form)  `(transfer-keyword-2-java-function ~k ~j-object ~@form)
     :else `(transfer-keyword-2-java-function ~k ~j-object ~form))))

(defmacro ->chain-calls
  "链式调用多个keyword到java函数的转换，keyword和参数必须一一对应，如果参数是个序列，需要将参数放在一个vector中（不用序列是因为：参数很可能是一个表达式，也会被检测成为一个序列，如果表达式做参数还需要包一层序列，故用vector），如(HashMap.)，如果直接作参数会被拆开，做参数是需要写成[(HashMap.)]；无参的函数与keyword的对应值为nil"
  [j-object & forms]
  {:pre [(even? (count forms))]}
  (let [forms  (map (fn [[k v]] `(keyword-2-java ~k ~v)) (partition 2 forms))]
    `(-> ~j-object ~@forms)))

(defmacro doto-keyword-2-java
  "doto的keyword-2-java实现版本"
  [j-object & forms]
  {:pre [(even? (count forms))]}
  (let [gx (gensym)]
    `(let [~gx ~j-object]
       ~@(map (fn [[k v]] `(keyword-2-java ~gx ~k ~v)) (partition 2 forms))
       ~gx)))

(defmacro init-object-&-run
  "初始化对象并允许其后的函数，第一个参数是调用的函数。\n第二个参数为类class，需要完整的路径，防止引用找不到相应的类。类可能有参（包括一个和多个参数），也可能无参。\n如果第三个参数为关键字，则class构建的时候为无参，否则opt1为class生成对象的函数名。如果类型为vector,则opt1的内容为class构建对象的多个参数，不是vector类型的话则表示class构建的对象只有1个参数，参数为opt1"
  [f clazz opt1 & opts]
  (let [clazz (if (class? clazz) clazz (resolve clazz))]
    (cond (keyword? opt1) `(~f (new ~clazz) ~opt1 ~@opts)
          (vector? opt1) `(~f (new ~clazz ~@opt1) ~@opts)
          :else `(~f (new ~clazz ~opt1) ~@opts))))
