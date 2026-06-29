# Clash Verge 全局扩展脚本教程

这份文档解释你现在用的 Clash Verge 全局扩展脚本。目标是让你能自己改规则，而不是每次靠复制整段脚本。

## 1. 这个脚本到底在改什么

Clash Verge 的订阅本质上是一份配置，里面主要有两块：

```yaml
proxy-groups:
  - name: 节点选择
    type: select
    proxies:
      - 香港 01
      - 台湾 02

rules:
  - DOMAIN-SUFFIX,google.com,节点选择
  - GEOIP,CN,DIRECT
  - MATCH,节点选择
```

全局扩展脚本就是在订阅配置加载后，用 JavaScript 改这份配置。

入口函数是：

```js
function main(config, profileName) {
  // config 是当前订阅配置对象
  // profileName 是当前订阅名称
  return config;
}
```

你在脚本里改 `config`，最后 `return config`，Clash Verge 就会使用修改后的配置。

## 2. 三种代理模式

脚本里有这一句：

```js
config.mode = "rule";
```

意思是强制使用规则模式。

常见模式：

```text
rule    规则模式，按 rules 从上到下匹配
global  全局模式，所有流量都走同一个策略组
direct  直连模式，所有流量都不走代理
```

你之前日志里出现：

```text
using GLOBAL
```

一般就是处于全局模式。想让国内直连、国外代理，就必须用 `rule`。

## 3. proxy-groups 是什么

`proxy-groups` 是策略组，也就是 Clash Verge 代理页里看到的那些组。

例如：

```js
upsertGroup({
  name: "🌍 国外流量",
  type: "select",
  proxies: [
    "🚀 国外自动",
    "🛠 国外手动",
    "DIRECT",
  ],
});
```

它的意思是创建一个叫 `🌍 国外流量` 的策略组，里面可以选：

```text
🚀 国外自动
🛠 国外手动
DIRECT
```

规则如果写：

```text
MATCH,🌍 国外流量
```

表示前面规则都没匹配到的流量，最后交给 `🌍 国外流量` 这个组处理。

## 4. 常见策略组类型

### select

手动选择。

```js
{
  name: "🛠 国外手动",
  type: "select",
  "include-all": true,
}
```

意思是创建一个手动选择组，并把当前订阅里的所有节点放进去。

适合你自己挑节点。

### url-test

自动测速选择。

```js
{
  name: "🚀 国外自动",
  type: "url-test",
  url: "https://www.gstatic.com/generate_204",
  interval: 300,
  tolerance: 80,
  timeout: 5000,
  "include-all": true,
}
```

含义：

```text
type: url-test        自动测速，选延迟低的节点
url                  用这个地址测速
interval: 300        每 300 秒测速一次
tolerance: 80        延迟差距小于 80ms 时不频繁切换
timeout: 5000        5 秒超时
include-all: true    把当前配置里的所有节点都加入这个组
```

### DIRECT

`DIRECT` 不是策略组，而是 Clash 内置目标，表示直连。

```text
GEOIP,CN,DIRECT
```

意思是中国大陆 IP 直连。

## 5. icon 和 emoji 的区别

你脚本里有：

```js
icon: "https://www.google.com/s2/favicons?sz=64&domain=chatgpt.com"
```

这个是策略组图标。代理页面的大卡片一般能显示。

但是首页下拉框不一定显示这个 `icon`，所以你看到别人下拉框有图标，很多其实是组名里直接带 emoji：

```js
const AI_GROUP = "🤖 ChatGPT";
const GLOBAL_GROUP = "🌍 国外流量";
```

结论：

```text
想让代理页卡片好看：加 icon 字段
想让下拉框也看见图标：组名里加 emoji
```

## 6. rules 是怎么匹配的

规则是从上往下匹配，匹配到第一条就停止。

例如：

```js
config.rules = [
  "DOMAIN-SUFFIX,chatgpt.com,🤖 ChatGPT",
  "GEOSITE,cn,DIRECT",
  "GEOIP,CN,DIRECT",
  "MATCH,🌍 国外流量",
];
```

访问 `chatgpt.com`：

```text
命中 DOMAIN-SUFFIX,chatgpt.com,🤖 ChatGPT
所以走 🤖 ChatGPT
```

访问 `baidu.com`：

```text
没有命中 chatgpt.com
命中 GEOSITE,cn,DIRECT
所以直连
```

访问 `google.com`：

```text
没有命中 chatgpt.com
不是国内
最后命中 MATCH,🌍 国外流量
所以走 🌍 国外流量
```

重点：`MATCH` 必须放最后。它是兜底规则，放前面会把后面的规则全部截胡。

## 7. 常见规则类型

### DOMAIN-SUFFIX

匹配域名后缀。

```text
DOMAIN-SUFFIX,openai.com,🤖 ChatGPT
```

能匹配：

```text
openai.com
api.openai.com
auth.openai.com
```

### DOMAIN

只匹配完整域名。

```text
DOMAIN,api.openai.com,🤖 ChatGPT
```

只匹配：

```text
api.openai.com
```

不匹配：

```text
chat.openai.com
```

### DOMAIN-KEYWORD

域名里包含关键词就匹配。

```text
DOMAIN-KEYWORD,bilibili,DIRECT
```

比较粗暴，可能误伤，但写起来方便。

### GEOSITE

按域名数据库匹配。

```text
GEOSITE,cn,DIRECT
```

表示中国大陆常见域名直连。

### GEOIP

按 IP 数据库匹配。

```text
GEOIP,CN,DIRECT
```

表示中国大陆 IP 直连。

### IP-CIDR

匹配 IP 段。

```text
IP-CIDR,192.168.0.0/16,DIRECT,no-resolve
```

表示局域网地址直连。

`no-resolve` 表示这条规则不触发 DNS 解析。

### MATCH

兜底规则。

```text
MATCH,🌍 国外流量
```

前面都没匹配到，就走这里。

## 8. 为什么之前分 AI 组和国外组

AI 当然也是国外流量，所以可以和普通国外流量共用一个组。

分开的原因是：ChatGPT/OpenAI 对节点质量、地区、风控更敏感。

比如：

```text
某个节点能打开 Google
但 ChatGPT 提示不可用、验证失败、频繁风控
```

如果 AI 和普通国外流量共用一个组，你只能整体换节点。

如果分开：

```text
ChatGPT -> 🤖 ChatGPT
Google/GitHub/YouTube -> 🌍 国外流量
```

你就可以只给 ChatGPT 换节点，不影响其他国外网站。

所以：

```text
想简单省事：一个组就够
想精细控制 ChatGPT：AI 单独一个组
```

## 9. 你这段脚本的结构

### 9.1 常量区

```js
const AI_GROUP = "🤖 ChatGPT";
const AI_AUTO = "⚡ AI自动";
const AI_MANUAL = "🧩 AI手动";
```

这些是组名。后面规则里必须完全一致，包括 emoji。

例如组名是：

```js
const AI_GROUP = "🤖 ChatGPT";
```

规则就必须写：

```js
`DOMAIN-SUFFIX,chatgpt.com,${AI_GROUP}`
```

不能写成：

```text
DOMAIN-SUFFIX,chatgpt.com,ChatGPT
```

因为 `ChatGPT` 和 `🤖 ChatGPT` 是两个不同名字。

### 9.2 清理旧组

```js
const oldGroupNames = [
  "ChatGPT", "AI自动", "AI手动",
  "国外流量", "国外自动", "国外手动",
  AI_GROUP, AI_AUTO, AI_MANUAL,
  GLOBAL_GROUP, GLOBAL_AUTO, GLOBAL_MANUAL,
];

config["proxy-groups"] = config["proxy-groups"].filter((g) => {
  return !oldGroupNames.includes(g.name);
});
```

作用：删掉以前脚本创建过的旧组，避免你改名后出现重复组。

比如你以前叫：

```text
ChatGPT
```

后来改成：

```text
🤖 ChatGPT
```

如果不清理，代理页可能同时出现两个组。

### 9.3 upsertGroup

```js
function upsertGroup(group) {
  const index = groups.findIndex((g) => g.name === group.name);
  if (index >= 0) {
    groups[index] = Object.assign({}, groups[index], group);
  } else {
    groups.unshift(group);
  }
}
```

作用：添加或更新策略组。

逻辑：

```text
如果已经有同名组：更新它
如果没有同名组：插到最前面
```

`groups.unshift(group)` 是插到数组开头，所以你新建的组会显示在代理页比较靠前的位置。

### 9.4 创建自动组

```js
upsertGroup({
  name: GLOBAL_AUTO,
  type: "url-test",
  url: "https://www.gstatic.com/generate_204",
  interval: 300,
  tolerance: 80,
  timeout: 5000,
  "include-all": true,
  icon: ICON_AUTO,
});
```

这是普通国外流量自动选择组。

`include-all: true` 表示把当前订阅配置里的所有节点加入这个组。

注意：如果你当前只选了一个机场订阅，它只包含这个机场的节点。想把多个机场节点放一起，需要先做多订阅合并。

### 9.5 创建手动组

```js
upsertGroup({
  name: GLOBAL_MANUAL,
  type: "select",
  "include-all": true,
  icon: ICON_MANUAL,
});
```

这是手动选择组。

当自动选择不满意时，可以手动挑节点。

### 9.6 创建入口组

```js
upsertGroup({
  name: GLOBAL_GROUP,
  type: "select",
  proxies: [
    GLOBAL_AUTO,
    GLOBAL_MANUAL,
    "DIRECT",
  ],
});
```

这是普通国外流量的总入口。

规则只需要指向它：

```text
MATCH,🌍 国外流量
```

你可以在这个组里选择：

```text
🚀 国外自动
🛠 国外手动
DIRECT
```

### 9.7 AI 规则

```js
const aiRules = [
  `DOMAIN-SUFFIX,chatgpt.com,${AI_GROUP}`,
  `DOMAIN-SUFFIX,openai.com,${AI_GROUP}`,
];
```

作用：把 ChatGPT/OpenAI 相关域名单独送到 AI 组。

如果你不想单独分 AI，可以删掉 AI 组，让这些域名最后通过 `MATCH` 走普通国外组。

### 9.8 国内直连规则

```js
const directRules = [
  "DOMAIN-SUFFIX,cn,DIRECT",
  "GEOSITE,cn,DIRECT",
  "GEOIP,CN,DIRECT",
];
```

作用：国内域名和国内 IP 直连。

建议 `GEOSITE,cn` 和 `GEOIP,CN` 都保留。

### 9.9 清理旧规则

```js
const oldRules = config.rules.filter((rule) => {
  const text = String(rule);

  for (const name of oldGroupNames) {
    if (text.includes(`,${name}`)) return false;
  }

  if (text.startsWith("MATCH,")) return false;

  return true;
});
```

作用：

```text
删掉旧脚本创建的规则
删掉原来的 MATCH 兜底规则
保留机场订阅自带的其他规则
```

为什么要删原来的 `MATCH`？

因为最后要重新加：

```js
`MATCH,${GLOBAL_GROUP}`
```

否则原机场配置里的 `MATCH` 可能提前截胡，导致你的兜底组不生效。

### 9.10 最终规则顺序

```js
config.rules = [
  ...aiRules,
  ...directRules,
  ...oldRules,
  `MATCH,${GLOBAL_GROUP}`,
];
```

含义：

```text
1. AI 域名先走 AI 组
2. 国内流量直连
3. 机场原本规则继续生效
4. 最后没匹配到的全部走国外流量
```

## 10. 最简单单组版

如果你不想区分 AI 和普通国外流量，可以用单组逻辑：

```text
国内 -> DIRECT
其他所有 -> 🚀 自动代理
```

核心脚本结构：

```js
function main(config, profileName) {
  const PROXY_GROUP = "🚀 自动代理";

  config.mode = "rule";
  config["proxy-groups"] = config["proxy-groups"] || [];
  config.rules = config.rules || [];

  config["proxy-groups"] = config["proxy-groups"].filter((g) => {
    return g.name !== PROXY_GROUP;
  });

  config["proxy-groups"].unshift({
    name: PROXY_GROUP,
    type: "url-test",
    url: "https://www.gstatic.com/generate_204",
    interval: 300,
    tolerance: 80,
    timeout: 5000,
    "include-all": true,
  });

  config.rules = [
    "DOMAIN-SUFFIX,local,DIRECT",
    "DOMAIN-SUFFIX,localhost,DIRECT",
    "DOMAIN-SUFFIX,cn,DIRECT",
    "GEOSITE,cn,DIRECT",
    "GEOIP,CN,DIRECT",
    `MATCH,${PROXY_GROUP}`,
  ];

  return config;
}
```

这个版本最简单，但不能给 ChatGPT 单独换节点。

## 11. 两组推荐版

如果你想兼顾简单和可控，推荐两组：

```text
🤖 ChatGPT    专门给 ChatGPT/OpenAI
🌍 国外流量   普通国外网站
```

规则顺序：

```text
ChatGPT/OpenAI 域名 -> 🤖 ChatGPT
国内域名/IP -> DIRECT
其他国外 -> 🌍 国外流量
```

这种方式比六个组更清楚。

具体可以这样设计：

```text
🤖 ChatGPT
  - ⚡ AI自动
  - 🧩 AI手动
  - 🌍 国外流量

🌍 国外流量
  - 🚀 国外自动
  - 🛠 国外手动
  - DIRECT
```

## 12. 怎么自己加一个网站规则

比如你想让 GitHub 明确走国外流量：

```js
const githubRules = [
  `DOMAIN-SUFFIX,github.com,${GLOBAL_GROUP}`,
  `DOMAIN-SUFFIX,githubusercontent.com,${GLOBAL_GROUP}`,
];
```

然后最终规则里加进去：

```js
config.rules = [
  ...aiRules,
  ...githubRules,
  ...directRules,
  ...oldRules,
  `MATCH,${GLOBAL_GROUP}`,
];
```

## 13. 怎么让某个网站直连

比如你想让学校网站直连：

```js
const directRules = [
  "DOMAIN-SUFFIX,sjtu.edu.cn,DIRECT",
  "GEOSITE,cn,DIRECT",
  "GEOIP,CN,DIRECT",
];
```

注意直连规则要放在 `MATCH` 前面。

## 14. 怎么让某个网站走指定组

比如想让 YouTube 走单独组：

```js
const YOUTUBE_GROUP = "📺 YouTube";
```

先建组：

```js
upsertGroup({
  name: YOUTUBE_GROUP,
  type: "select",
  "include-all": true,
});
```

再加规则：

```js
const youtubeRules = [
  `DOMAIN-SUFFIX,youtube.com,${YOUTUBE_GROUP}`,
  `DOMAIN-SUFFIX,googlevideo.com,${YOUTUBE_GROUP}`,
  `DOMAIN-SUFFIX,ytimg.com,${YOUTUBE_GROUP}`,
];
```

最后：

```js
config.rules = [
  ...youtubeRules,
  ...directRules,
  ...oldRules,
  `MATCH,${GLOBAL_GROUP}`,
];
```

## 15. 常见问题

### 为什么日志还是 using GLOBAL

检查是否还是全局模式。

必须是：

```text
代理模式：规则
```

脚本里也要有：

```js
config.mode = "rule";
```

### 为什么下拉框没有 icon

首页下拉框不一定显示 `icon` 字段。想显示，组名里加 emoji。

### 为什么 AI手动 显示 Timeout

手动组本身不是测速组，显示 Timeout 不一定代表不能用。展开后手动选具体节点再看。

### 为什么多个机场节点没有都出现

`include-all: true` 只包含当前配置里的节点。

如果多个机场是多个独立订阅卡片，当前只启用了其中一个，那脚本只能看到这一个订阅的节点。

想所有机场一起自动选择，需要做多订阅合并配置，把多个订阅合成一个配置。

### 为什么国内网站也走代理

可能原因：

```text
规则模式没开
DIRECT 规则顺序太靠后
订阅原规则里有更早的代理规则
DNS 或 TUN 设置导致识别异常
```

优先看日志，确认命中了哪条规则。

### 为什么规则不生效

检查：

```text
组名是否完全一致，包括 emoji
MATCH 是否在最后
脚本是否保存
内核是否重启/配置是否重新加载
规则模式是否打开
```

## 16. 推荐你怎么写

如果你想省心：

```text
国内 DIRECT
国外 🚀 自动代理
```

用单组。

如果你常用 ChatGPT：

```text
ChatGPT/OpenAI -> 🤖 ChatGPT
国内 -> DIRECT
其他国外 -> 🌍 国外流量
```

用两组。

如果你想精细控制 YouTube、Netflix、Telegram、GitHub：

```text
每类服务一个组
每类服务一组规则
最后 MATCH 到默认国外组
```

但组越多，越难维护。一般两组就够。

