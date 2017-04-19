# dl4clj-example

本项目是deeplearning4j示例的clojure版本。

## 项目结构
本项目主要分为dl4j示例代码的clojure化和dl4j的clojure封装。
示例代码在dl4clj_example目录，对应于dl4j代码的org.deeplearing4j.example目录。
封装代码在deeplearning4clj目录，对应于dl4j代码的org.deeplearing4j目录。
本项是计划一般编写示例代码，一边完善封装代码。并计划随代码的完善，最终将封装代码独立处理作为一个项目。

## 面向数据的编程
在本项目中，作者尽可能对程序的实现细节进行封装，是使用者能够专注于数据和参数而不是编程。
在本项目中，作者尽力使输入的数据格式都保持为clojure的基本数据。如：根据传统方式的话，在构建dl4j的神经网络的时候需要大量的java函数，无论是代码风格还是整体结构都比较繁杂。作者通过clojure宏实现了一个键值对方式的结构，类似于clojure的Map数据类型。

## 吃你的狗粮，让你无粮可吃
本项目的希望告别的“吃自己的狗粮（Eating your own dog food）”，实现“吃你的狗粮，让你无粮可吃”。
