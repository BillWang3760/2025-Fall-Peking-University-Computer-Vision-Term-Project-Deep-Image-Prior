# 2025-Fall-Peking-University-Computer-Vision-Term-Project-Deep-Image-Prior

欢迎来到我们的Repository！  

本仓库提供25秋北京大学信息科学技术学院《计算机视觉》课程大作业（选题为Deep Image Prior）的可运行代码与报告，以下将主要介绍代码的有关情况，希望帮助减轻助教的工作量并为此后选课的同学进行复现提供参考。

## deep-image-prior

这一部分代码直接基于Ulyanov的代码库：

https://github.com/DmitryUlyanov/deep-image-prior

虽然原代码库的热度不低，但由于年久失修的原因，其pytorch=0.4等低配置已经使得该代码库很难像当初那样plug and play了。对此，我们采取了一些可控的补救措施；
现在你所看到的代码可以直接在Google Colab上运行，不需要其他任何配置。我们也准备了可以在本地运行的版本，如果你确有需要，请联系我们。

这一部分代码还增加了多尺度DIP图象重绘的实现，这是我们工作中的一个创新点，希望你喜欢。
