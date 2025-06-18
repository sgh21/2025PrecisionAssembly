> 本文介绍基于Windows系统如何快速环境搭建，内容分为两部分，第一部分为编辑器VScode和环境管理工具Anaconda的介绍和配置方法，第二部分为针对本次比赛的Python环境搭建
>

# 开发工具和基础环境
## VScode
### 简介
正如我们写文档需要Word此类编辑器一样，VScode本质上也是一种编辑器，我们需要用它来编写我们的代码。之所以选用VScode而不用PyCharm此类“编译器”，是因为VScode相对来说更为轻量，基本功能相对简洁，因此上手难度低。但其简单而不简陋，VScode凭借强大的插件功能，使其能够胜任各种工作场景和编程语言，并且其基于配置文件的UI和环境配置使得其更具有定制化，可玩性高，并且学习曲线平滑，熟练掌握VScode的使用不仅可以提高开发效率，也能使得开发者对于程序的运行逻辑和底层原理有更深刻的认识，这是PyCharm此类集成开发环境（IDE）无法获得的。

### 安装包下载和配置
1. 在浏览器搜索VScode或者点击下面连接，即可找到VScode的官网，点击Download for Windows进行安装包下载

[Visual Studio Code - Code Editing. Redefined](https://code.visualstudio.com/)

2. 与其它软件安装无异，只需双击安装包，修改安装路径后安装即可，比较简单，不再赘述。
3. 在安装完成之后，需要下载基本插件来进行Python代码的编写，使得编辑器知道，所有以.py为结尾的文件都是Python文件，并能正确的识别它。操作步骤为，打开VScode左侧拓展商店，需要安装Python插件，其图标样式如下。

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750083172767-939615f7-813a-46c9-9a2a-d8e697e3cc53.png)

4. 至此，VScode的基本配置就完成了，其还有其它丰富的插件，比较常用的有Chinese(翻译插件)，以及比较好用的Github Copilot（集成在VScode中的AI帮写，但需要Github账号进行学生认证才可免费使用，感兴趣可以自己探索下，下面连接是参考教程）

[学生认证](https://zhuanlan.zhihu.com/p/29735660108)

## Python基础环境
### 简介
在VScode中，我们完成了编辑器的安装，学过C++的同学应该知道，编程语言运行时需要将面向开发者的变成语言编译成面向计算机的二进制机器码，因此编写C++需要依赖GNC这样的编译器。Python与C++有所不同，Python是解释性语言，无需编译，但依然需要对编程语言进行转译，让计算机可以读懂，因此我们需要安装Python解释器，或者Python基础环境。

### Python包下载和安装
1. 在浏览器搜索Python或者点击下面连接，即可找到Python的官网，点击Download，选择Python版本，推使用Python3.7.8，这是由于傲博官方提供的SDK（软件开发工具包）只支持python3.7版本。建议直接下载可执行安装包，比较方便。

[Python Releases for Windows](https://www.python.org/downloads/windows/)

2. 安装时勾选将Python加入环境变量，也就是让你的电脑的其它软件（主要是VScode）能够找到你的Python解释器。然后选择自定义安装，确保pip工具包为勾选状态，pip是库管理工具，我们可以用它快速下载别的开发者上传的Python包。如下图所示

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750084234014-066ffb2a-fee9-41aa-914a-f79644720cbc.png)

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750084216717-636f43be-507a-432b-ac89-efd7e30589d5.png)

3. 安装完成后，打开终端，输入python，会进入python环境，并显示当前版本信息。打印无错说明Python解释器安装正确，并且环境变量配置无误，如果有问题请检查这两步。输入exit()推出python环境。环境变量的更改需要重启计算机，如果输出不正确重启一下试试吧。

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750084380117-eff191a6-b712-49e7-8d48-b04619931ceb.png)

## Anaconda
### 简介
我们在进行开发工作时，是站在巨人的肩膀上的，而非从零开始，因此我们往往要下载一些成熟的开发包。但是在各种项目开发过程中，我们会使用到各种各样的包，不同项目要求的包的版本可能不兼容，因此，我们每进行一个项目开发时，都希望能够有独立的环境，这样方便管理。Anaconda就是一款成熟的Python环境管理软件。

### 下载和安装
1. 下载都大同小异了，相信大家已经掌握了，下面是下载链接。

[Download Success | Anaconda](https://www.anaconda.com/download-success)

2. 显然，Anaconda要管理我们的电脑环境，自然也需要将其加入环境变量，使得我们的终端能够知道它的位置，因此，安装时需要勾选将Anaconda加入环境变量

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750084928765-d1307729-f469-43a8-89ec-487bae651f03.png)

3. 验证安装成功的方式相同，在终端输入conda，查看有无软件信息日志输出

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750085010954-7ba90a55-ca8a-4784-8978-487eb6e110e7.png)

4. 基本的安装步骤就是这样，如果遇到什么问题，以及Anaconda的基础使用可以参考以下文章

[请进行安全验证(Security Verification)](https://blog.csdn.net/Natsuago/article/details/143081283)

### 基本使用
+ Anaconda常见的命令其实很简单，分别为创建环境，激活（进入）环境，退出环境，安装Python包和删除环境
+ 创建环境并下载制定版本Python解释器

```bash
conda create -n env_name python==3.7.8
```

+ 激活环境

```bash
conda activate env_name
```

+ 退出环境

```bash
conda deactivate
```

+ 安装指定Python包

```bash
conda install package_name
```

+ 删除环境

```bash
conda remove -n env_name --all
```

# 本项目Python包配置
## 硬件SDK配置
### 简介
本次比赛涉及的硬件系统有Aubo5i六轴机械臂，MVS海康工业单目相机，以及个人PC。我们在使用个人电脑对这些硬件进行控制时需要调用官方提供的软件开发接口（函数），因此，我们需要事先配置相关包。但其配置相对简单，在提供的代码框架中已经完成相关配置，并对硬件接口进行了封装，给出了方便调用的函数接口。

### 配置介绍
```bash
├─config
├─documents
├─lib
│  ├─Aubo5i
│  │  
│  └─MvImport   
├─logfiles
├─models
├─src
└─utils
```

在lib文件夹下的两个文件夹即为对应的硬件SDK功能包，分别为官方提供的Aubo5i，控制接口和MVS海康相机接口，其具体实现无需掌握。在utils文件夹下，分别实现了AuboControlLowLevel.py和MVSControl.py两个函数，对SDK进行了进一步封装，方便调用。在使用时请参考文件最后的使用示例，以及函数的注释，内容比较细碎，不再展开，不过功能都比较容易理解，实在不懂的可以进一步私下交流。

## 软件Python包配置
### 简介
本项目依赖了一些第三方的软件包，基于一些成熟的工具快速实现软件开发，主要包括numpy(科学计算包），opencv（视觉处理包）和scipy（主要应用了其多种角度表示转换）

### 安装方法
比较通用的方法是先创建一个Anaconda环境，指定Python解释器版本为3.7.8，然后使用pip安装指定包，具体流程如下

```bash
conda create -n aubo3.7.8 python==3.7.8 					 #创建conda环境
conda activate aubo3.7.8													 #激活conda环境
pip install numpy, opencv-python==4.10.0.84, scipy #安装指定包
```

我们的项目中也会使用到Yolo（成熟的图像识别和分割算法），受制于Python版本，我们需要安装较低的Yolov8版本，因此安装稍有一些复杂，并且依赖Git。接下来将进行Yolo本地安装介绍

1. 安装Git，Git是代码托管工具，方便我们管理不同的代码版本，也用于代码共享和协作，其安装方式不是很复杂，请参考以下文章。

[请进行安全验证(Security Verification)](https://blog.csdn.net/weixin_42242910/article/details/136297201)

2. 随后，使用Git下载Yolo的远程包，在终端使用以下命令。

```bash
git clone https://github.com/ultralytics/ultralytics
```

3. 我们需要安装指定的版本，因此使用如下命令切换到指定Tag

```bash
git checkout -b yolov8 v8.0.100
```

4. 然后在先前创建的conda环境下，使用pip进行yolo的安装，终端所在路径为yolo的主文件夹，终端环境应为aubo3.7.8，命令如下

```bash
pip install -e .
```

终端情景如图所示：

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750087479767-eb577629-8888-4766-9f62-4acfd64a1b8b.png)

5. 验证是否安装成功，可以使用如下命令查看已经安装的Python包，查看是否有ultralytics

```bash
pip list
```

![](https://cdn.nlark.com/yuque/0/2025/png/46299634/1750087649124-6e12f170-76d9-4fe0-ba58-ba2edb1034d0.png)

6. 如有其它问题可参考如下文章中源码安装部分，或与我私下讨论

[请进行安全验证(Security Verification)](https://blog.csdn.net/weixin_45819759/article/details/131962654)

---

> 至此我们的环境就完全搭建好了，可以运行MVSControl.py和AuboControlLowLevel.py进行测试，如有其它任何问题，欢迎与我交流
>

