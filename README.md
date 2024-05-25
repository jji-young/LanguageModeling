# LanguageModeling

##  24-1학기 인공신경망과 딥러닝 Assingment #3

## 1. Plot of the average loss values for training and validation.

: CharRNN, CharLSTM 둘 다 Epoch의 증가에 따라 Train Loss와 Validation Loss가 곡선을 그리며 감소하는 경향을 보임. 모델이 안정적으로 학습된 것을 확인할 수 있음

+ ### VanilaRNN 
    <img src="img\rnn.png"> 
+ ### LSTM 
    <img src="img\lstm.png">
: CharRNN, CharLSTM 의 validation loss 값 (Epoch = 30 기준) 

|               |CharRNN|CharLSTM|
|---------------|------|-----|
| Cross Entropy Loss(Train)|1.4073|1.1640|
| Cross Entropy Loss(Valid)|1.4136|1.1826|


## 2. Difference in text generation based on temperature parameter T & Seed Character
|Temperature |Seed|Text by Generating RNN|Text by Generating LSTM|
|---------------|------|-----|-----|
|0.5|A|at I will they should shall the other am and therefore you shall not her more and her true the suady |air action of the world We have been a world of heaven, If you thinks to drown the man I can the peop | B |but that the hath botten that will be such and you have preturn to be deterson,As much to such or he |
| | E | e and once the hearts,Which thy king of his name to your grace to stand of the worlds of entreat wit  | es of spoken and her brother Gloucester, And will not a dangerous ladies, And not how consul, be in t
| | K |king; for the country: when it, my lord, and you shall the supreme the father to hear Citizen:You sa   |ks, see him be speaks to make my heart Than a prove my country's grace words: 'eath, and your grace t
| | P | peak.First Senator:It and be in the way the people,And to the tribunes As I have strong of my goo |pon their mother, south, you have some tomune of the war, to his helpless, and hear not to the Tower,
| 1.0 | A | an men in the streely.  BUCKINGHAM: Let us briefly grace theak'tise devil! No men to maile.  PRINCE E|aving the liture is given to make my looks are he returng, and so do I say 's death.' Of the nobility|
|   |B|belived against the further. I did as rashedge? or do't. His moroother bring the devil.  DERTARENBMy|be sons shall be streed To thus for time mayor, Your good for's seen your lament and Dron with or bra|
|   |E|e!  QUEEN MARGARET: Those alled fault our lord of what proceoces find?  Second Come, I: 'Tis senators|e honour than my field Cominius ways.  GREY: Nothin, for he says When he spire speed A brow are you w|
|   |K|ke did first tell threep about to adventy by a keeditious leatios, let your such's need, who.  BUCKIN|k are before. Nay, dis; Tarpest of Dees: I will not were, in life?  BRUTUS: Margey find Untoul after|
|   |P|prook to shall none of Freathely pluck noble: Faked other's majesty Releating heall on, Such a good r|power than spirit, after many seen Abend as power not, I say! I'll be denow their venture fittes! I p |
|1.5|A|ay my names you hard, say my, fliend perfalley!  GLOUCESTER: No, you frate,'d slum'? the boral. Telns|ann; all, no who, sow catchbosn of thy Volsnee I wis, Caur gentle: Ho! foo the '? Should yourse, who |
|   |B|birush Fied iuthose a him, and fow thee  Second wilm; well ranglliven vaith ruthliate'd, re usturggen |boy Of han, chilt two fear them nolook'd your wits, Who, heral wife. O, ripe hife hire surm-ifie Tisp |
|   |E|es has anonou me.  QUEEN ELIZABETH: Geal'd? Yow' togury: Hark I anforawn my lickers, well meit.- On w|efores But no, no abunnesy, scorn so well writtle. O thandagisting, which by you: you gons: Ho lalian |
|   |K|krication? 'pley took my sathims ouch vined.' Nort, But,, her wife! One aisoncuse. What he city? a sw |k with it: on 't is your--  MARCIUS: How dost spirites, yet thou with Yout soothe, I cannot purpose.|
|   |P|proud upons Foll to'e unvouts, is And long well with thy gals. Uponance you says moscies conuot! Dy g |pon our grace that's privy seates And act not a pitious, anodgm on--Phokf Place; Manished'd me but te|

**: Temperature의 변화에 따라 생성된 Text 를 살펴보면,**

**Temperature = 0.5일 때,**  비교적 자연스러운 단어와 문장구조로 문장이 생성됨

**Temperature  = 1.0 일 때,** 간혹 존재하지 않는 단어들이 등장하지만, 과반수 이상이 자연스러운 단어들로 문장이 생성되었으며 문장 구조가 부자연스러운 경향이 있음

**Temperature  = 1.5 일 때,** 존재하지 않는 단어들의 비율이 매우 높으며, 제대로된 문장의 형태를 띄고있다고 보기 어려움

+ Softmax function with a Temperature parameter $T$ 는 다음과 같음

$$
y_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

위 수식에 따르면, T값에 따라 생성되는 Text는 다음과 같이 예측 할 수 있음

**$T<1:$** : 생성될 확률이 높은 문자의 확률이 더욱 높아짐

**$T=1:$** : 기존의 softmax 확률과 동일

**$T<1:$** : 문자들이 생성될 확률이 모두 같아짐


## 3. Discussion
+ RNN의 장기 의존성 문제를 해결하기 위해 등장한 LSTM이 이번 Shakespeare Language modeling 에서도 RNN보다 더욱 우수한 성능을 보임 
+ **parameter $T$** 의 변화에 따라 생성되는 Text에서, $T=0.5$ 일 때도 다소 부자연스러운 Text들이 등장하는 것을 보며, Langauage modeling에서 RNN, LSTM의 한계점을 느낌
+ 현재 대중적으로 사용되고 있는 LLM의 attention 연산 등 최신 기법의 적용되었을때의 생성되는 Text 구현에도 도전해보고 싶다는 생각을 갖게 됨




