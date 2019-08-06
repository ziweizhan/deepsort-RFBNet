# deepsort-RFBNet
# 视频效果：https://www.bilibili.com/video/av62427886/
# 安装教程：
##  1. 配置环境
  conda env create -f RFBNet.yml  
##  2. 首先配置RFBNet，直接按照RFBNet的官方教程编译一下：  
  ./make.sh  
##  3. （如果2编译成功忽略3）如果RFBNet编译出现问题，请重新下载RFBNet，按照官方教程编译就可以了。
     编译完成之后将本工程中的RFBNet/test_net.py替换编译好的RFBNet就可以了。  
##  4. 下载model
下载RFBnet的检测model放在weights文件夹里面，默认的model是 voc-300  下载地址：https://pan.baidu.com/s/1xOp3_FDk49YlJ-6C-xQfHw  
##  5. 运行 
python rfbnet.py     
##  6. 其它问题
      注意rfbnet.py里面的路径需要设置成你自己当前的路径。  
      import sys  sys.path.append('/home/你的路径/RFBNet-deep-sort/RFBNet')
