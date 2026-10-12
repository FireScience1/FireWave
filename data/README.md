# FIT 数据

FireWave 使用 FIT（Fire in Tunnel）数据目录，目录结构应保持为：

```text
FDS_fire/
  A1/ ... A4/
  B1/ ... B4/
  ...
```

每个场景至少需要温度、能见度和热释放率文件；代码通过
`firewave.core.FITStore` 读取。默认配置中的 `data_root` 是
`../FDS_fire`，也可以在运行时替换为绝对路径或 HDFS 挂载路径。

The complete FIT archive is tracked with Git LFS in this repository. A regular
Git clone retrieves lightweight pointer files; install Git LFS and run
`git lfs pull` to download the actual data before running the models.
