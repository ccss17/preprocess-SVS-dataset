# Preprocessing SVS Dataset

preprocessing code for SVS dataset with [midii](https://github.com/ccss17/midii)

## Installation

```shell
pip install git+https://github.com/ccss17/preprocess-SVS-dataset.git
```

## Usage

```python
import preprocess_svs as ps
```

# TODO

- 앞뒤 공백음 제거
- lyric 이 없으면 pitch 를 0 으로 만들고 공백음으로 병합
- 0.3초 이하 공백음 제거하고 전 또는 후 노트의 duration 을 그만큼 늘려주기
- class 화