
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


model_name = "/home/demo/.cache/kagglehub/models/mistral-ai/mistral/pyTorch/7b-instruct-v0.1-hf/1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

data = """
[
    {
      "metadata": {
        "page_number": "2",
        "file_name": "keph204.pdf"
      },
      "context": "THERMODYNAMICS 227\nThermodynamics is the branch of physics that\ndeals with the concepts of heat and temperature\nand the inter -conversion of heat and other for ms\nof energy. Thermodynamics is a macroscopic\nscience. It deals with bulk systems and does not\ngo into the molecular constitution of matter . In\nfact, its concepts and laws were formulated in the\nnineteenth century before the molecular picture\nof matter was firmly established.  Thermodynamic\ndescription involves relatively few macroscopic\nvariables of the system, which are suggested by\ncommon sense and can be usually measured\ndirectly. A microscopic description of a gas, for\nexample, would involve specifying the co-ordinates\nand velocities of the huge number of molecules\nconstituting the gas. The description in kinetic\ntheory of gases is not so detailed but it does involve\nmolecular distribution of velocities.\nThermodynamic description of a gas, on the other\nhand, avoids the molecular description altogether .",
      "summary": "THERMODYNAMICS 227\nThermodynamics is the branch of physics that\ndeals with the concepts of heat and temperature\nand the inter -conversion of heat and other for ms\nof energy. Thermodynamics is a macroscopic\nscience."
    },
    {
      "metadata": {
        "page_number": "164",
        "file_name": "kech105.pdf"
      },
      "context": "chemIstry164\nsUmmary\nThermodynamics deals with energy changes in chemical or physical processes and enables us to \nstudy these changes quantitatively and to make useful predictions. For these purposes, we divide the universe into the system and the surroundings. Chemical or physical processes lead to evolution or absorption of heat (q), part of which may be converted into work (w). These quantities are related through the first law of thermodynamics  via ∆U = q + w. ∆U,  change in internal energy, depends \non initial and final states only and is a state function, whereas q and w depend on the path and \nare not the state functions. We follow sign conventions of q and w by giving the positive sign to \nthese quantities when these are added to the system. We can measure the transfer of heat from one system to another which causes the change in temperature. The magnitude of rise in temperature depends on the heat capacity (C) of a substance. Therefore, heat absorbed or evolved is q =  C∆T.",
      "summary": "chemIstry164\nsUmmary\nThermodynamics deals with energy changes in chemical or physical processes and enables us to \nstudy these changes quantitatively and to make useful predictions. For these purposes, we divide the universe into the system and the surroundings."
    },
    {
      "metadata": {
        "page_number": "15",
        "file_name": "keph204.pdf"
      },
      "context": "PHYSICS 240\n3.The first law of thermodynamics is the general law of conservation of energy applied to\nany system in which energy transfer from or to the surroundings (through heat and\nwork) is taken into account.  It states that\n∆Q  = ∆U  +  ∆W\nwhere ∆Q is the heat supplied to the system, ∆W is the work done by the system and ∆U\nis the change in internal energy of the system.\n4.The specific heat capacity of a substance is defined by\nsmQ\nT=1∆\n∆\nwhere m is the mass of the substance and ∆Q is the heat required to change its\ntemperature by ∆T.  The molar specific heat capacity of a substance is defined by\n1QCTµ∆=∆\nwhere µ is the number  of moles of the substance. For a solid, the law of equipartition\nof energy gives\nC  =  3 R\nwhich generally agrees with experiment at ordinary temperatures.\nCalorie is the old unit of heat. 1 calorie is the amount of heat required to raise the\ntemperature of 1 g of water from 14.5 °C to 15.5 °C.  1 cal  =  4.186 J.",
      "summary": "PHYSICS 240\n3.The first law of thermodynamics is the general law of conservation of energy applied to\nany system in which energy transfer from or to the surroundings (through heat and\nwork) is taken into account. It states that\n∆Q  = ∆U  +  ∆W\nwhere ∆Q is the heat supplied to the system, ∆W is the work done by the system and ∆U\nis the change in internal energy of the system."
    }
  ]
"""
# # prompt = f"summarise the contents in the following data {data}"
query = f"what is thermodynamics"

json_resp = {
    "metadata": {
        "page_number": "all page_numbers",
        "file_name": "filenames"
    },
    "summary": "consolidated summary generated"
}
prompt = f"The provided data {data} conatins metadata and texts, your objective: \
    use metadata and text to genetate a response in json containing the following \
    [INST] {json_resp} [/INST]. Make sure that if the {query} does not relate to given context \
    give an empty json response."


sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)

output = sequences[0]['generated_text']
print(output.split("Answer:"))
# print(output.split())


# print(sequences[0]['generated_text'])