# Server Log 查看指南（非 root 账号）

适用场景：华为云 ECS 上，使用非 `root` 账号查看后端服务日志。

---

## 方案 A：读取日志文件（推荐）

### 1) 创建只读运维用户

```bash
sudo adduser ttavops
```

### 2) 创建日志组并加用户

```bash
sudo groupadd ttavlog
sudo usermod -aG ttavlog ttavops
```

### 3) 授权日志文件给日志组（组可读）

```bash
sudo chgrp ttavlog /var/log/ttav-backend.log
sudo chmod 640 /var/log/ttav-backend.log
```

### 4) 确保日志轮转后权限不丢

在该日志对应的 `logrotate` 配置中加入（或确认存在）：

```conf
create 0640 root ttavlog
```

### 5) （可选）ACL 保底，避免应用重建日志导致权限回退

```bash
sudo setfacl -m g:ttavlog:r /var/log/ttav-backend.log
```

### 6) 用户重新登录后查看日志

```bash
tail -F /var/log/ttav-backend.log
```

说明：建议使用 `-F`，比 `-f` 更稳，日志轮转后会自动跟踪新文件。

---

## 方案 B：如果主要使用 systemd/journald

如果服务日志主要在 journal 中，可以不开放文件权限，直接授予 journal 读取权限：

```bash
sudo usermod -aG systemd-journal ttavops
```

重新登录后查看：

```bash
journalctl -u ttav-backend -f
```

---

## 排查小贴士

1. 用户加组后必须重新登录（或新开 SSH 会话）才会生效。  
2. 如果 `tail -F` 报权限错误，先检查：
   - 文件组是否是 `ttavlog`
   - 文件权限是否是 `640`
   - 用户是否在 `ttavlog` 组内（`id ttavops`）
3. 如果日志是应用自行创建，优先在应用日志配置中固定文件属组/权限，ACL 作为补充。
