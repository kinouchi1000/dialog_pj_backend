# dialog project backend

NTT の japanese dialog transformer を用いた、対話システムのプロジェクト

# requirement

-   docker
-   python

# Quick start

1. Download the NTT dialog model

    You can download from [japanese-dilaog-transformers](https://github.com/nttcslab/japanese-dialog-transformers#model-download)
    Please store model to `docker/inference_server/model`

    ```bash
    cd docker/infenrence_server/model
    wget https://www.dropbox.com/s/k3ugxmr7nw6t86l/japanese-dialog-transformer-1.6B.pt
    ```

    If you download the different model like [this](https://www.dropbox.com/s/e5ib6rhsbldup3v/japanese-dialog-transformer-1.6B-persona50k.pt?dl=0), please edit common/constants.py

2. Download the sentencepiece dictionaries

    Clone [this project](https://github.com/nttcslab/japanese-dialog-transformers) and, you can find the dictionary and some file in `data/`.
    Please store them to `docker/inference_server/data`

    ```bash
    cd docker/infenrence_server/data
    git clone https://github.com/nttcslab/japanese-dialog-transformers
    mv japanese-dialog-transformers/data ./
    rm -rf japanese-dialog-transformers
    ```

3. Let's build image and run container

    ```bash
    cd docker
    docker compose build
    docker compose up -d
    ```

    You can see the logs with below.

    ```bash
    docker compose logs -f
    ```

4. Install some libraries

    ```bash
    cd ../
    pip install -r requirement.txt
    ```

5. Please run test code

    ```bash
    python test/client_sample/main.py
    ```

    You can see result like below.

    ```bash
    ❯ python test/client_sample/main.py

    ```

    you can see like below. Note that you can stop dialog with `/stop`

    ```bash
    >>こんにちは。よろしくね。
    こんにちは!こちらこそよろしくお願いします。お仕事は何をされてるんですか?
    >>お仕事はプログラマーです。
    かっこいいですね!わたしはキャリアコンサルタントをしています。趣味は読書なんですが、あなたは何か趣味はありますか?
    >>/stop
    ['かっこいいですね!わたしはキャリアコンサルタントをしています。趣味は読書なんですが、あなたは何か趣味はありますか?']
    ['こんにちは。よろしくね。', 'こんにちは!こちらこそよろしくお願いします。お仕事は何をされてるんですか?', 'お仕事はプログラマーです。', 'かっこいいですね!わたしはキャリアコンサルタントをしています。趣味は読書なんですが、あなたは何か趣味はありますか?']
    ```

# Test dialog.py in local environment

```bash
 cd src
python src/inference_server/model/dialog.py
```
