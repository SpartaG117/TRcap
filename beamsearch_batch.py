import torch.nn as nn

class Beamsearch(object):
    def __init__(self, beam_size, batch_size, max_len):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.log_softmax.cuda()
        self.beams = [[{'word': [1], 'logprob': 0.0}] for _ in range(batch_size)]
        self.final_beams = [[] for _ in range(batch_size)]
        self.current_len = 1

    def calc_num_beam(self, beams):
        beam_len = []
        for beam in beams:
            beam_len.append(len(beam))
        return beam_len

    def get_input_cap(self):
        """
        :return: a list of shape [batch_size x beam_size, current_len]
        """
        captions = []
        for i in range(self.batch_size):
            cnt = self.beam_size
            for beam in self.beams[i]:
                captions.append(beam['word'])
                cnt -= 1
            for _ in range(cnt):
                cap_pad = [0] * self.current_len
                captions.append(cap_pad)
        return captions

    def search(self, pred):
        # pred: a float tensor with shape [batch_size x (beam_size x current_len), num_vocab]
        cap_len = self.current_len

        # [batch_size, beam_size, num_vocab]
        outputs = self.log_softmax(pred).view(self.batch_size, self.beam_size, cap_len, -1)[:,:,-1,:].tolist()
        log_probs = []

        beam_len = self.calc_num_beam(self.beams)
        for b in range(self.batch_size):
            log_prob = []
            for i in range(beam_len[b]):
                log_prob.append(outputs[b][i])
            log_probs.append(log_prob)
        # log_probs is a list with shape [batch_size, num_beam, num_vocab]

        candidates = []
        for b in range(self.batch_size):
            candidate = []
            for i, word_logprobs in enumerate(log_probs[b]):
                sorted_word = [item[0] for item in sorted(enumerate(word_logprobs),
                                                          key=lambda x:x[1], reverse=True)]
                for word in sorted_word[:self.beam_size]:
                    candidate.append({
                        'word':self.beams[b][i]['word'] + [word],
                        'logprob': self.beams[b][i]['logprob'] + word_logprobs[word]
                    })

            candidate = sorted(candidate, key=lambda x:x['logprob'], reverse=True)[:self.beam_size]
            candidates.append(candidate)

        remain_candidates = []
        for b in range(self.batch_size):
            remain_candidate = []
            final_beam = []
            for candidate in candidates[b]:
                # 2 is the id of </S>
                if candidate['word'][-1] == 2:
                    self.final_beams[b].append(candidate)
                elif len(candidate['word']) == self.max_len:
                    self.final_beams[b].append(candidate)
                else:
                    remain_candidate.append(candidate)

            remain_candidates.append(remain_candidate)

        self.beams = remain_candidates
        self.current_len += 1

        beam_len = self.calc_num_beam(self.beams)
        flag = 0
        for b in range(self.batch_size):
            if beam_len[b] != 0:
                flag = 1

        return flag

    def get_results(self):
        beam_len = self.calc_num_beam(self.final_beams)

        final_cap = []
        for b in range(self.batch_size):
            if beam_len[b] != 0:
                cap_sorted = sorted(self.final_beams[b], key=lambda x:x['logprob'],
                                    reverse=True)
                # if final word is </S>
                if cap_sorted[0]['word'][-1] == 2:
                    final_cap.append(cap_sorted[0]['word'][1:-1])
                else:
                    final_cap.append(cap_sorted[0]['word'][1:])
        return final_cap


if __name__ == '__main__':
    searcher = Beamsearch(3,10)
    print(searcher.beams)