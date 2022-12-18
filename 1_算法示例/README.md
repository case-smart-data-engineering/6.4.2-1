# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
4. 在程序运行开始后，需要在命令行中选择输入参数，参数包括train（训练）和test（测试），即在命令行中
   输入 `train或test` ,然后按 `ENTER` 继续运行示例程序。
   注：应先输入 `train` 进行训练，会自动在该目录下生成模型文件，获得模型后，重新运行程序后，输入`test`
      进行测试，在测试阶段需要输入要翻译的中文序列，然后按 `ENTER` 继续运行，获得翻译后的英文序列。
   示例：
      #运行程序后,命令窗口中的输出如下：
      Number of samples: xx
      Number of unique input tokens: xx
      Number of unique output tokens: xx
      Max sequence length for inputs: xx
      Max sequence length for outputs: xx
      select train model or test model:       #在此处选择输入参数train/test
      #应先选择train，进行模型训练
      #训练完成后，程序运行结束
      #重新执行程序运行步骤，此时参数选择为test，进行预测
      #测试时，命令窗口有如下输出：
      请输入要翻译的中文:                       #在此处输入要翻译的中文序列
      （提示：因为数据集太小，训练的模型在做测试时效果不好，故在输入要翻译的中文时应选择数据集中给出的句子，例如：难以置信、我爱中国等
             若要翻译的是数据集外的句子，则会结束循环，程序运行结束）
      ......



注：
数据集：
中英文对照表，文本文件，每一行代表一个数据项，每个数据项由中文+TAB+英文组成。
从中获取中文作为输入序列，对应的英文作为输出序列。
从数据集中选取百分之十作为验证集。

