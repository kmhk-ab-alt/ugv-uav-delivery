# UGV・UAV協調配送シミュレーション

ROS 2 JazzyとGazebo Harmonicを用いて、無人地上車両（UGV）と無人航空機（UAV）が協力して荷物を配送するシミュレーションを構築した。UGVは集荷センターで荷物を受け取り、道路で到達できる配送先へ運ぶ。山間部などへの配送では、合流地点で荷物をUAVに受け渡し、UAVが配送先まで空輸する。

配送全体の管理は、司令センターに相当するソフトウェアが担う。カメラ画像、注文情報、UGV・UAVの稼働状況を集約し、道路の通行状況の認識、配送する荷物と担当機体の選択、合流地点の選定、経路計画、移動制御を連携させる。ブラウザーでは、機体の動作と、その動作につながった判断を確認できる。

## 主な構成要素と役割

| 構成要素 | 役割 |
|---|---|
| VLM | Gazeboの仮想カメラ画像から道路の通行可否と荷物の重要度ラベルを認識する。また、注文情報に基づく配送の優先方針を、判断根拠とともに提示する。 |
| OpenRouter | VLMへのアクセスに用いるAPIサービス。起動スクリプトではNVIDIA **Nemotron 3 Nano Omni**を指定している。モデルIDは `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`。 |
| 配車MLP（Multilayer Perceptron：多層パーセプトロン） | 配送可能な荷物の中から、運ぶ荷物と担当機体を選択する。候補は道路の通行状況と機体の稼働状況に基づいて絞り込み、P3、P2、P1の順で最も重要度の高いグループをMLPに渡す。学習済みの重みは `checkpoints/site_dispatch/best.npz` に格納している。 |
| 合流地点の選定 | UGV・UAVの予測到着時間と合流地点の予約状況を比較し、荷物の受け渡しに適した地点を選択する。 |
| 経路計画・移動制御 | 通行状況と障害物の位置を考慮して経路を計画する。機体の位置・速度を取得し、目標に沿って走行・飛行・停止するように速度指令を生成する。 |

### VLMの判断基準と認識のタイミング

VLMに与える判断基準は、`pai_ros/site_perception.py` の `JUDGMENT_PROMPT` に定義している。道路ごとに画像を評価し、建物の屋根や影、路面の模様と、通行を妨げる障害物を区別するように指示する。荷物の重要度はP3を最も高く設定し、この優先順位はVLMの回答によって変更しない。

道路の画像認識は、**シミュレーションの起動時と、あらかじめ設定された障害物の配置変更時**に行う。その都度カメラ画像を取得し、OpenRouter経由でVLMに送信して通行可否を判定する。障害物の配置変更は再認識を開始するきっかけとして扱い、判定そのものには画像を用いる。画像を常時監視して環境の変化を検出する方式ではない。

道路認識の結果が返るまではシミュレーションを一時停止し、結果を配送計画に反映してから処理を再開する。荷物については、一度取得した認識結果を同じ実行中に再利用し、同じ荷物に対するAPIへの重複した問い合わせを抑制する。

## 起動手順

同梱の環境定義はApple Silicon Mac（`osx-arm64`）を対象としている。実行にはPixiとXcode Command Line Toolsが必要となる。Pixiのインストール方法は[公式ドキュメント](https://pixi.sh/latest/installation/)を参照する。

### 事前準備：環境構築とビルド

リポジトリを取得した後、そのフォルダーに移動して次のコマンドを実行する。

```bash
pixi install --locked
bash build.bash
export PAI_WORKSPACE="$PWD"
```

`build.bash` は、荷物の保持・解放を行うGazeboプラグインとROS 2パッケージをビルドする。シミュレーションの起動時には、`install` フォルダーに配置されたコードを読み込む。各スクリプトは自身の保存場所を基準に必要なファイルを参照するため、スクリプト内のパスを書き換える必要はない。

`PAI_WORKSPACE` は、以降のコマンドで使用するリポジトリの保存先を表す。別のターミナルで作業を続ける場合は、リポジトリのフォルダーに移動し、`export PAI_WORKSPACE="$PWD"` を再度実行する。

### 1. OpenRouterのAPIキーを設定する

OpenRouterでAPIキーを発行する。次のコマンドを実行し、入力を求められたらキーを貼り付けてEnterを押す。入力した文字は画面に表示されない。この操作は初回、またはキーを変更するときに行う。

```bash
bash -c '
umask 077
mkdir -p "$PAI_WORKSPACE/runtime"
read -rsp "OpenRouter API key: " key
printf "\n"
printf "%s" "$key" > "$PAI_WORKSPACE/runtime/openrouter-api-key"
chmod 600 "$PAI_WORKSPACE/runtime/openrouter-api-key"
unset key
'
```

キーは `runtime/openrouter-api-key` に保存する。このファイルはGitの管理対象から除外している。環境変数 `OPENROUTER_API_KEY` にもキーが設定されている場合は、環境変数の値を優先して使用する。

### 2. OpenRouterのVLMを使って起動する

```bash
bash "$PAI_WORKSPACE/run-showcase-vlm.bash" --showcase-hold 90
```

起動すると、ブラウザーにシミュレーション画面が表示される。自動で開かない場合は、ターミナルに `Presentation: http://127.0.0.1:xxxxx/` と表示されたURLを開く。ポート番号は起動ごとに変わるため、その都度表示されたURLを使用する。

- 終了するには、起動に使用したターミナルで **Ctrl+C** を押す。
- 再起動するには、Ctrl+Cで終了してから起動コマンドを再度実行する。ソースコードを変更した場合は、再ビルドしてから起動する。
- `--headless` を付けると、ブラウザーの自動起動を無効にできる。画面配信用のHTTPサーバーは起動するため、表示されたURLからアクセスできる。
- `--showcase-hold 90` は、シミュレーション終了後も画面を90秒間表示する設定である。
- `--model モデルID` を付けると、使用するVLMを変更できる。変更する際は、OpenRouterで対象モデルの画像入力への対応状況と利用料金を確認する。

### 3. OpenRouterを使わずにデモを起動する

```bash
bash "$PAI_WORKSPACE/run-showcase.bash" --showcase-hold 90
```

APIを使用せずに動作を確認する場合は、こちらの起動スクリプトを使用する。画像認識を合成画像とルールに基づく処理に置き換えるため、APIキーは不要となる。画面には「VLM未使用」と表示される。

### 実行時間を指定する

新規配送を開始できる時間は、`--operation-limit 秒数` で指定する。省略時はシミュレーション内の180秒（3分）となる。5分に設定する場合は、次のように起動する。

```bash
bash "$PAI_WORKSPACE/run-showcase-vlm.bash" --operation-limit 300 --showcase-hold 90
```

10分に設定する場合は `--operation-limit 600` を指定する。この制限はシミュレーション内の経過時間に適用され、VLMの応答待ちなどで一時停止している時間は含まない。制限時間に達すると新規の配車を終了するが、開始済みの配送と帰還は継続する。

実時間での実行上限は、別途 `--timeout 秒数` で指定する。既定値は1200秒（20分）である。長時間のシミュレーションを実行する場合は、必要に応じてこの上限も延長する。

## 設定とログ

起動コマンドの末尾にオプションを追加すると、以下の設定を変更できる。

| オプション | 内容 |
|---|---|
| `--operation-limit 秒数` | 新規配送を開始できる時間を、シミュレーション内の秒数で指定する。 |
| `--speed-scale 倍率` | UGV・UAVの巡航速度を調整する。 |
| `--site-events off` | 障害物の配置変更を無効にする。道路の画像認識は起動時の1回だけ行う。 |
| `--perception pixels` | 荷物の重要度ラベルを、APIを使わずローカルで判定する。道路の認識方法は変更しない。 |
| `--record-dir 保存先` | 認識結果や配車判断の記録先を指定する。 |
| `--headless` | ブラウザーの自動起動を無効にする。 |

利用可能なオプションは、ビルド後に次のコマンドで確認できる。

```bash
bash "$PAI_WORKSPACE/run-gazebo.bash" --help
```

実行設定とGazeboのログは `runtime/gazebo_runs/` に、ROSのログは `runtime/ros/log/` に保存される。OpenRouterを使用する起動スクリプトでは、認識結果と配車判断を `runtime/recordings/` に記録する。これらの保存先は、いずれもリポジトリのフォルダーを基準としている。APIキーと実行ログはGitの管理対象から除外している。

## ソース変更後のビルド

稼働中のシミュレーションを終了し、リポジトリのフォルダーで再ビルドする。

```bash
cd "$PAI_WORKSPACE"
bash build.bash
```

Pythonのテストは、リポジトリのフォルダーで実行する。bashを使用し、実行前に環境を有効化する。

```bash
source "$PAI_WORKSPACE/activate.bash"
python -m unittest discover -s test
```

ブラウザー画面のテストにはNode.jsを使用する。Node.jsを用意したうえで、同じフォルダーから次のコマンドを実行する。

```bash
node test/test_story.mjs
```
