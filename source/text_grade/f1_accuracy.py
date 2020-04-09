# %%

# %%

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import pickle
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt


EPOCHES = 1001


def load_data(file_path):
    x = pickle.load(open(file_path, 'rb+'))
    return x

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "./drive1/My Drive/grade/checkpoints/train/"
test_log_dir = "./drive1/My Drive/grade/checkpoints/train"
train_summary_writer = tf.compat.v2.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.compat.v2.summary.create_file_writer(test_log_dir)
Buffer_size = 2000
BATCH_SIZE = 64
Max_length = 200
Score_dataset = load_data('./drive/My Drive/data/group_split/train_1_score.ds')
Article_dataset = load_data('./drive/My Drive/data/group_split/train_1_article.ds')
#Group_dataset = load_data('./drive/My Drive/data/test_and_train/train_group.ds')


Score_dataset = np.array(Score_dataset)
Article_dataset = np.array(Article_dataset)
#Group_dataset = np.array(Group_dataset)
Score_dataset = Score_dataset.astype(np.int64)
Article_dataset = Article_dataset.astype(np.float32)
#Group_dataset = Group_dataset.astype(np.int64)
train_data = tf.data.Dataset.from_tensor_slices((Score_dataset, Article_dataset))
Score_dataset = load_data('./drive/My Drive/data/group_split/test_1_score.ds')
Article_dataset = load_data('./drive/My Drive/data/group_split/test_1_article.ds')
#Group_dataset = load_data('./drive/My Drive/data/test_and_train/test_group.ds')
Score_dataset = np.array(Score_dataset)
Article_dataset = np.array(Article_dataset)
#Group_dataset = np.array(Group_dataset)
Score_dataset = Score_dataset.astype(np.int64)
Article_dataset = Article_dataset.astype(np.float32)
#Group_dataset = Group_dataset.astype(np.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((Score_dataset, Article_dataset))

'''
#测试加载函数功能以及函数转换功能
for (a,(b,c,d)) in enumerate(train_data):
    b=tf.strings.to_number(b,tf.int64)
    print(a,b,c,d)
    break
'''

# %%


train_data = train_data.shuffle(Buffer_size)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([],[-1, -1]))
test_data = test_dataset.shuffle(Buffer_size)
test_dataset = test_data.padded_batch(BATCH_SIZE, padded_shapes=([], [-1, -1]))


# %%

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# %%

'''
#这里仅仅是用来测试
pos_encoding = positional_encoding(50, 512)


plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
'''


# %%

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# %%

def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # print(matmul_qk.shape)
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # print('scaled',scaled_attention_logits.shape)
    # 将 mask 加入到缩放的张量上。
    # if mask is not None:
    #  scaled_attention_logits += (mask * -1e9)
    # print('mask',mask.shape)
    # print('scaled2',scaled_attention_logits.shape)
    ## softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    # print(attention_weights.shape)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    # print('out',output.shape)
    return output, attention_weights


# %%

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=tf.keras.initializers.glorot_normal,
                                        bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=tf.keras.initializers.glorot_normal,
                                           bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


np.set_printoptions(suppress=True)


# %%


# %%


# %%


# %%

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal,
                              bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_initializer=tf.keras.initializers.glorot_normal,
                              bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        # (batch_size, seq_len, d_model)
    ])


# %%

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# %%

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(300,d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        # print('encoder shape',x.shape)
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# %%


# %%

def loss_function(pred, real):

    loss_ = loss_object(real, pred)


    return tf.reduce_mean(loss_)
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.next_layer = tf.keras.layers.Dense(1024, kernel_initializer='random_uniform',
                                               bias_initializer='zeros',
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.final_layer = tf.keras.layers.Dense(1, kernel_initializer='random_uniform',
                                                 bias_initializer='zeros',
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        enc_output = tf.reshape(enc_output, [-1, 96 * d_model])
        #fin_out = self.next_layer(enc_output))
        enc_output2 = self.final_layer(enc_output)
        #enc_output2 = tf.nn.softmax(enc_output2, axis=-1)
        return enc_output2


# %%

num_layers = 2
d_model = 300
dff = 2048
num_heads = 6
dropout_rate = 0.1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# %%

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
'''
#这里是为了显示激活函数
temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
'''

#损失函数使用记录：第一次MeanabsoluteError，效果不好
train_loss = tf.keras.metrics.Mean(name='train_loss')
group_loss = tf.keras.metrics.Mean(name='group_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy()
loss_object=tf.keras.losses.MeanSquaredError()
train_accuracy=tf.keras.metrics.Accuracy()



def create_masks(inp):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    return enc_padding_mask


transformer = Transformer(num_layers, d_model, num_heads, dff, pe_input=200)

# %%

checkpoint_path = "./drive/My Drive/grade/checkpoints"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
#if ckpt_manager.latest_checkpoint:
#    ckpt.restore(ckpt_manager.latest_checkpoint)
#    print('Latest checkpoint restored!!')

# %%

EPOCHES = 1001

tf.compat.v2.summary.trace_on(graph=True, profiler=False)
# %%

train_step_signature = [
    tf.TensorSpec(shape=(None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None),dtype=tf.float32)
]
loss_result=[]
@tf.function(input_signature=train_step_signature)
def train_step(score, article,real_score):
    enc_padding_mask = create_masks(article[:][:][1])
    print(article)
    with tf.GradientTape() as tape:
        predictions = transformer(article, True, enc_padding_mask)

        loss = loss_function(predictions, score)
        #loss2 = tf.reduce_mean(group_loss_object(group, group_predict))
        #loss = loss1 + loss2

    train_loss(loss)
    #predictions=predictions+0.5
    #predictions=tf.cast(predictions,tf.int64)
    score=tf.cast(score,tf.float32)
    predictions=tf.reshape(predictions,[-1])
    predictions=predictions-score
    predictions=tf.abs(predictions)
    predictions=tf.cast(tf.greater(predictions,1),tf.int64)

    train_accuracy.update_state(real_score,predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))


for epoch in range(EPOCHES):
    start = time.time()
    # inp -> portuguese, tar -> english
    loss_record=[]
    train_accuracy.reset_states()
    train_loss.reset_states()
    real=tf.zeros([1])


    for (batch, (score, article)) in enumerate(train_data):

        train_step(score,article,real)
    if (epoch + 1) % 5 == 0:
       ckpt_save_path = ckpt_manager.save()
       print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
    with train_summary_writer.as_default():
        tf.compat.v2.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.compat.v2.summary.scalar('train_translage_accruacy', train_accuracy.result(), step=epoch)

    print('Epoch {} Loss {:.4f} Accuracy {:.4f} '.format(epoch + 1,
                                                                                               train_loss.result(),
                                                                                               train_accuracy.result(),
                                                                                             ))
    train_accuracy.reset_states()
    train_loss.reset_states()

    for (batch, (score,article)) in enumerate(test_dataset):
        # enc_padding_mask=create_masks(article[:][:][1])
        predictions= transformer(article, True, 0)
        loss = loss_function(predictions,score)
        train_loss(loss)

        predictions=tf.reshape(predictions,predictions.shape.as_list()[0])
        '''
        predictions=predictions+0.5
        predictions=tf.cast(predictions,tf.int64)
        train_accuracy(score,predictions)
        '''
        score=tf.cast(score,tf.float32)
        predictions=tf.subtract(predictions,score)

        predictions=tf.abs(predictions)
        predictions=tf.cast(tf.greater(predictions,1),tf.int64)
        real_s=tf.zeros([predictions.shape.as_list()[0]])
        train_accuracy(real_s,predictions)

    with test_summary_writer.as_default():
        tf.compat.v2.summary.scalar('test_loss', train_loss.result(), step=epoch)
        tf.compat.v2.summary.scalar('test_translage_accruacy',train_accuracy.result(), step=epoch)
    print('Epoch {} test_Loss {:.4f} test_Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

# %%


# %%


# %%


# %%


# %%



