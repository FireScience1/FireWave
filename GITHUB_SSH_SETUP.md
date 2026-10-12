# GitHub SSH 配置

本项目仓库：<https://github.com/FireScience1/FireWave>

- SSH key 名称：`cli`
- 类型：Authentication Key（Read/write）
- 指纹：`SHA256:y6DwyLKHHET6bvFRZLjYRjN3HsEpRW1bQEK28uStFkU`
- 本地公钥文件：`/home/tiger/.ssh/id_rsa.pub`

公钥内容（仅公钥，不包含私钥）：

```text
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDHABNECqGcTI3n9IyZs8uCWeZNEJ2gfRb+DR+UAZ646xrPMeQXJtqSEZrm2RFvizjV1X618iZ5CZ+5ooylb+xV5wnZQ8nKDco/H2AaS2siFqDYWfkAUKCtwgXyweJBbAOHYusPAwz9RuoF9i/XDj/yLrL71wra4XjR78F5+FQaDnQ7nQ23s4YUw9LDr37cVCi7FnNKFid11vnVR5E6FghO0EDEQivz7fJYJOr0ejthtm2FbqO7oipZiHw/5wXBGqMl/7JeqQnijr8iDAwRXBR7ZQkhuNo9eoTIYqsIeVvCPLcAjNHYLSk1BbuE12UC5rJQ16j/+3bybODTTk+VRM/iBBZ32/1qjetshjX5PqTAyb0W8sprdtJkCqld/URKYo4zHqhIARUBDZX2aNQfq7cJByGeTkkT0A5m4qx01lOEA93b/q1hl8gmzfy/8E3nJJwmPwD+AwH2AiyfxgPtKzaVoR/RKH356sNdd/stzS/Xkh4J36ZU7kgvIzEcjzZVqSstx7/m8b1987p4Z9o2YuvwMKl2snYRb6sokybjMC4xkBrl2h29Knzd7vky1LfUw5zoRvDEAygqi1DnRaJm5mJU0729wwrcd0Wm+wZmFtQzY2YatN/iMdi4DvJJ4t88pRq4OakJZiIu6WIre5UKZwhdzsIo2zNPK2z2SmgyRFtPQZQ== tiger@mlxlabj1t6sk396aaa35d8-20260916062320-vfftu8-master
```

## SSH 远程地址

```bash
cd /mlx_devbox/users/zuowei.zhang/playground/fire4/FireWave
git remote set-url origin git@github.com:FireScience1/FireWave.git
ssh -T git@github.com
```

认证成功时会显示：`Hi FireScience1! You've successfully authenticated, but GitHub does not provide shell access.`
