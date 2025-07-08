from src.haze.core import Haze
from src.haze.models import IdeaModel, DecoderModel
from src.encoder.core import NumericEncoder
from src.decoder.core import ArgMax
from src.dataset.core import generate_binary_sequence

def main():
    haze = Haze(sequential=True, persist=False)
    haze.load()
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=ArgMax(), outputs=[0,1])]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    row_size = 8
    dataset_size = 10000
    dataset = generate_binary_sequence(dataset_size=dataset_size, row_size=row_size)
    reward_scoring = []
    for row in dataset:
        haze.observe(input_data=row, encoder=lexical_set[0].encoders[0])
        decoder_id = lexical_set[0].decoders[0].decoder.get_id(as_string=True)
        prediction = haze.predict(row_size)
        result = prediction[decoder_id]
        reward = 0
        if len(result) == row_size:
            for answer, problem in zip(row, result):
                if answer == problem:
                    reward += 1
        reward = reward/row_size
        print(row)
        print(result)
        print(reward)
        reward_scoring.append(reward)
        haze.learn(reward=reward)

    print(reward_scoring)

if __name__ == '__main__':
    main()