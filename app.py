import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


st.set_page_config(layout="wide")


@st.cache_resource()
def initialize_models():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    tokenizer_ft = AutoTokenizer.from_pretrained("model-finetuned/epoch_2")
    model_ft = AutoModelForCausalLM.from_pretrained("model-finetuned/epoch_2")  

    return tokenizer, model, tokenizer_ft, model_ft

tokenizer_gpt2, model_gpt2, tokenizer_gpt2_ft, model_gpt2_ft = initialize_models()


@st.cache_resource()
def load_recipes():
    return pd.read_pickle('dataset-prepared/df_test_display.pkl')


def encode_promt(promt_text, tokenizer):
    encoded_promt = tokenizer.encode(
        promt_text, 
        add_special_tokens=False, 
        return_tensors='pt'
    )
    return encoded_promt


def create_output_sequences(model, encoded_promt, num_sequences=3):
    output_sequences = model.generate(
        input_ids=encoded_promt,
        max_length=600,
        temperature=0.9,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=num_sequences,
        early_stopping=True
    )

    return output_sequences


def process_output_sequences(output_sequences, tokenizer, with_special_token=False):
    special_token = "<|endoftext|>"
    results = []

    for i, output_sequence in enumerate(output_sequences):
        result = tokenizer.decode(output_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if with_special_token:
            results.append(result[:result.index(special_token)])
        else:
           results.append(result) 

    return results 


def format_output_sequences_to_html(decoded_output_sequences):
    output = []

    for out_seq in decoded_output_sequences:
        instructions = out_seq.split('Instructions:')[1]
        steps = instructions.split('\n')
        steps = [step.strip().capitalize() for step in steps if len(step.strip()) > 2]
        formatted_steps = ''
        for i, step in enumerate(steps):
            formatted_steps += f'<p>{i+1}.\n {step}</p>'
        output.append(formatted_steps)
    return  output


def initalize_state():
    st.session_state.ingredients = ''
    st.session_state.recipe_selected = ''
    st.session_state.prepared_ingredient_list = ''
    st.session_state.finetuned_responses = []
    st.session_state.base_model_responses = []


def suggest_recipe(show_base_model=False):
    promt_text = f"""
    Give me a recipe which contains only the listed ingredients.
    Ingredients:
    {st.session_state.prepared_ingredient_list}
    Instructions:
    """

    # fine tuned model
    encoded_promt = encode_promt(promt_text, tokenizer_gpt2_ft)
    output_sequences = create_output_sequences(model_gpt2_ft, encoded_promt, num_sequences=3)
    decoded_output_sequences = process_output_sequences(output_sequences, tokenizer_gpt2_ft)
    st.session_state.finetuned_responses = format_output_sequences_to_html(decoded_output_sequences)
    # base model
    if show_base_model:
        encoded_promt_base = encode_promt(promt_text, tokenizer_gpt2)
        output_sequences_base = create_output_sequences(model_gpt2, encoded_promt_base, num_sequences=3)
        decoded_output_sequences_base = process_output_sequences(output_sequences_base, tokenizer_gpt2, with_special_token=False)
        st.session_state.base_model_responses = format_output_sequences_to_html(decoded_output_sequences_base)


def render_btns(ui_element):
    if ui_element.button('Suggest Recipe'):
        suggest_recipe()


def get_entry_val():
    return st.session_state.ingredients if len(st.session_state.ingredients) > 4 else ''


def render_app():
    initalize_state()
    recipes = load_recipes()


    col_m_1, col_m_2, col_m_3 = st.columns([1,7,1])
    col_m_2.markdown('<div style="display:flex;flex-direction: row;justify-content: space-between;"><span style="margin: auto 0;font-size:2em;font-weight:bold;"><img height="40" src="https://www.shareicon.net/data/128x128/2016/08/19/817046_food_512x512.png" style="margin-bottom:10px;"/> <span>lunch with me</span></span><span style="font-size:6em;"><span>🥡</span></span></div>', unsafe_allow_html=True)
    col_m1_1, col_m1_2 = col_m_2.columns(2)


    st.session_state.recipe_selected = col_m1_2.selectbox(
        'Choose ingredients from a selection of recipes:',
        ('select one of the recipes', *recipes['title'].values))


    st.session_state.ingredients = col_m1_2.text_area('Enter ingredients:', placeholder='Enter ingredients - separated by comma', value=get_entry_val())

    prepared_ingredients = ''
    predicted_instructions = '' 

    if st.session_state.recipe_selected != 'select one of the recipes':
        vals = recipes[recipes['title'] ==  st.session_state.recipe_selected]

        for ing in vals['ingredients'].values[0]:
            prepared_ingredients += f'<div>{ing.strip()}</div>'
            
        for i, inst in enumerate(vals['instructions'].values[0]):
            predicted_instructions += f'<div>{i+1}. {inst.strip().capitalize()}</div>'

        st.session_state.prepared_ingredient_list = '\n'.join(vals['ingredients'].values[0])

    if st.session_state.ingredients:
        for ing in st.session_state.ingredients.split(','):
            prepared_ingredients += f'<div>{ing.strip()}</div>'




    col_m1_1.markdown(f'<div class="hero-card"><img width="100%" height="100%" style="border-radius:8px;" src="https://images.kitchenstories.io/wagtailOriginalImages/A866-photo-final-01.jpg"/></div>', unsafe_allow_html=True)

    render_btns(col_m1_2)

    col_n_1, col_n_2, col_n_3 = st.columns([1,7,1])
    col_n1_1, col_n1_2 = col_n_2.columns(2)

    if prepared_ingredients:
        col_n1_1.markdown(f'<div class="hero-card" style="padding:18px"><div style="font-size:1.2em;font-weight:600;">Selected ingredients:</div><div>{prepared_ingredients}</div></div>', unsafe_allow_html=True) 

    if predicted_instructions:
        col_n1_2.markdown(f'<div class="hero-card" style="padding:18px"><div style="font-size:1.2em;font-weight:600;">Original recipe steps:</div><div >{predicted_instructions}</div></div>', unsafe_allow_html=True)  

    col_m_2.write('')
    col_m_2.write('')
    col_m_2.write('')
    if st.session_state.finetuned_responses:
        col_n_2.subheader('🤗 GPT2 distilled - finetuned')
        col_1, col_2, col_3 = col_n_2.columns(3)
    
        col_1.markdown(f"<div class='custom-card'><div>{st.session_state.finetuned_responses[0]}</div></div>", unsafe_allow_html=True)
        col_2.markdown(f'<div class="custom-card"><div>{st.session_state.finetuned_responses[1]}</div></div>', unsafe_allow_html=True) 
        col_3.markdown(f'<div class="custom-card"><div>{st.session_state.finetuned_responses[2]}</div></div>', unsafe_allow_html=True) 


    if st.session_state.base_model_responses:
        col_n_2.subheader('🤗 GPT2 distilled')
        col_a, col_b, col_c = col_n_2.columns(3)
        col_a.markdown(f"<div class='custom-card'><div>{st.session_state.base_model_responses[0]}</div></div>", unsafe_allow_html=True)
        col_b.markdown(f'<div class="custom-card"><div>{st.session_state.base_model_responses[1]}</div></div>', unsafe_allow_html=True) 
        col_c.markdown(f'<div class="custom-card"><div>{st.session_state.base_model_responses[2]}</div></div>', unsafe_allow_html=True) 

    render_styles()
    add_bg_from_url() 


def render_styles():
    style = """
    .hero-card {
        box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
        border-radius: 8px;
        margin: 8px 0;

    }
    .ingredients-card {
    padding: 12px;
    background: linear-gradient(
        rgba(255,255,255,.9), 
        rgba(255,255,255,.8)), 
        url('https://cdn.vox-cdn.com/thumbor/ytEw8jLppkl3U2y5vvtJ1CmurEs=/0x0:1920x1080/1200x800/filters:focal(807x387:1113x693)/cdn.vox-cdn.com/uploads/chorus_image/image/66136952/177483_sashimistylewatermelonpokebowlveggiepretlandscape_618639.0.jpg');
        background-size: contain;
    }
    .main-divider {
        background: linear-gradient(
            rgba(255,255,255,0), 
            rgba(255,255,255,0.3)), 
            url('https://i.pinimg.com/originals/22/fc/33/22fc33caf6707869016fe0c8e6ec8475.jpg');
        background-size: center;

        border-radius: 8px;
    }
    .custom-card {
        min-height: 200px;
        padding: 12px;
        border-radius: 8px;
        background-color: #fff;
        color: black;
    }
    .custom-card > div {
        position: relative;
        z-index: 10 !important;
    }
    div.stButton > button:first-child {
        background-color: black;
        color:#ffffff;
        font-size: 36px;
        padding: 16px 46px;
        margin: auto 0;
    }
    div.stButton > button:first-child > p {
        font-size: 36px;
    }
    div.stButton > button:hover {
        background-color: #E57373;
        color:#fff;
    }
    div.stButton > button:focus:not(:active) {
        background-color: #E57373;
        color: #fff;
        border:none;
    }

    .custom-card:before {
        content: "";
        z-index: 0;
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        background: linear-gradient(-45deg, #fbe9d7 0%, #f6d5f7 100% );
        transform: translate3d(0px, 20px, 0) scale(0.95);
        filter: blur(20px);
        opacity: var(0.7);
        transition: opacity 0.3s;
        border-radius: inherit;
    }

    .custom-card::after {
        content: "";
        z-index: 0;
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        background: inherit;
        border-radius: inherit;
    }
    """
    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(255,255,255,0.99), 
                rgba(255, 255, 255, 0.98)),
                url('https://theveganatlas.com/wp-content/uploads/2020/05/Hawaiian-Vegan-Poke-bowl2.jpg');
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    render_app()