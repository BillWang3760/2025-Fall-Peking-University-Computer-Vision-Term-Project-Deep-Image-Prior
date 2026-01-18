# 2025-Fall-Peking-University-Computer-Vision-Term-Project-Deep-Image-Prior

欢迎来到我们的Repository！  

本仓库提供25秋北京大学信息科学技术学院《计算机视觉》课程大作业（选题为Deep Image Prior）的可运行代码与报告，以下将主要介绍代码的有关情况，希望帮助减轻助教的工作量并为此后选课的同学进行复现提供参考。

## deep-image-prior

这一部分代码直接基于Ulyanov的代码库：

https://github.com/DmitryUlyanov/deep-image-prior

虽然原代码库的热度不低，但由于年久失修的原因，其pytorch=0.4等低配置已经使得该代码库很难像当初那样plug and play了。对此，我们采取了一些可控的补救措施；
现在你所看到的代码可以直接在Google Colab上运行，不需要其他任何配置。我们也准备了可以在本地运行的版本，如果你确有需要，请联系我们。

这一部分代码还增加了多尺度DIP图象重绘的实现，这是我们工作中的一个创新点，希望你喜欢。

## DIP blending

打开deep-image-prior后最上方的文件夹即为DIP blending。它的运行环境与本课程的Homework 5 / Homework 6是相同的，运行时请把models和utils拷贝到该文件夹中。

这是我们工作中的另一个创新点。@liuzhonghe060718为多尺度DIP图像融合技术的设计、实现与实验付出了宝贵的心血，它在该类型任务中的表现超过了我们在Homework 2中所实现的Alpha blending / Poisson blending / Laplacian blending，这真的很不容易。尽管DIP本身已经很难说是图像融合技术方面的主流技术，但是如果你对此————特别是对多尺度DIP图像融合技术————仍然抱有兴趣，欢迎与我们取得联系。
