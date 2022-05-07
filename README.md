# dialog project backend

NTT の japanese dialog transformer を用いた、対話システムのプロジェクト

# requirement

-   docker
-   python

# Quick start

1. Let's build image and run container

    ```bash
    cd docker
    docker compose build
    docker compose up -d
    ```

    You can see the logs with below.

    ```bash
    docker compose logs -f
    ```

2. Install some libraries

    ```bash
    cd ../
    pip install -r requirement.txt
    ```

3. Please run test code

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

# dialog.py を試運転

```bash
 cd src
python src/inference_server/model/dialog.py
```
