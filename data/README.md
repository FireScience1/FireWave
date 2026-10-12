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

原始 FIT 数据约 4.2 GB，因此不纳入普通 Git 提交；请从项目数据存储复制到
本目录对应位置后再运行配置。仓库只保留数据布局和读取约定，避免把大文件或
本地绝对路径提交到 GitHub。
