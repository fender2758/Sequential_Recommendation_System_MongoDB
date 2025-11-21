import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
from io import BytesIO
from pymongo import MongoClient
import time
import pandas as pd
import numpy as np
import torch

from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, PreTrainedTokenizerFast, BertConfig
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json

client = MongoClient(host='localhost', port=27017)
db = client['beauty']
collection = db['products']
reviews = db['reviews']


class ShoppingApp:
    def __init__(self, root, user_id):
        self.root = root
        self.asin_map = pd.read_csv('data/asin_mapping.csv')
        self.asin_map.set_index('asin', inplace=True)
        self.map_asin = pd.read_csv('data/asin_mapping.csv')
        self.map_asin.set_index('asin_numeric', inplace=True)

        self.root.title("Shopping Site")
        self.current_page = 0
        self.items_per_page = 10
        self.search_results = []  # Holds search results
        self.showing_default = True  # Track whether displaying default items
        self.user_id = user_id  # User ID passed from login page
        self.user_history = self.get_history(self.user_id)
          # Fetch default items
        
        # Logout Button
        self.logout_button = ttk.Button(root, text="Logout", command=self.logout)
        self.logout_button.pack(side="top", anchor="e", padx=10, pady=10)

        # Search bar, Home button, and Search button
        self.search_frame = tk.Frame(root)
        self.search_frame.pack(pady=10)

        self.search_entry = ttk.Entry(self.search_frame, width=50)
        self.search_entry.pack(side="left", padx=5)

        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.perform_search)
        self.search_button.pack(side="left", padx=5)

        self.home_button = ttk.Button(self.search_frame, text="Home", command=self.show_home)
        self.home_button.pack(side="left", padx=5)

        self.reset_button = ttk.Button(self.search_frame, text="Reset", command=self.reset_recommender)
        self.reset_button.pack(side="left", padx=5)

        # Create a scrollable canvas for products
        self.canvas = tk.Canvas(root, width=1080, height=720)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Pagination Buttons
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack()

        self.prev_button = ttk.Button(self.nav_frame, text="Previous", command=self.show_previous_page)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.show_next_page)
        self.next_button.grid(row=0, column=1, padx=10)
        self.tokenizer, self.model = self.get_model()
        
        self.recommended_items = self.get_recommended()
        # Display default items on load
        self.display_items(self.recommended_items)

    def logout(self):
        """Handle user logout and return to the login page."""
        self.root.destroy()  # Close the shopping app window
        main()  # Restart the application to show the login page

    def get_model(self):
            
        with open('vocab.json', 'r') as file:
            vocab = json.load(file)

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        tokenizer_file = "custom_tokenizer.json"
        tokenizer.save(tokenizer_file)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, unk_token="[UNK]", max_length=128)

        special_tokens = {'pad_token': '[PAD]', 'mask_token': '[MASK]'}
        tokenizer.add_special_tokens(special_tokens)


        # Define the model configuration
        config = BertConfig(
                vocab_size = len(tokenizer),
                hidden_size = 64,
                max_position_embeddings=100,
                attention_probs_dropout_prob=0.2,
                hidden_act='gelu',
                hidden_dropout_prob=0.2,
                initializer_range=0.02,
                num_attention_heads=2,
                num_hidden_layers=2,
                type_vocab_size=2,
        )

        model = BertForMaskedLM.from_pretrained('models/bert-item-mlm', config=config).to('cuda')
        return tokenizer, model

    def get_recommended(self):
        item_input = 'item' + ' item'.join([str(self.asin_map.loc[asin]['asin_numeric']) for asin in self.user_history])
        print(self.user_history)
        print(item_input)
        inputs = self.tokenizer(item_input, 
            max_length=100,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        try:
            idx = np.where(inputs['input_ids']==self.tokenizer.pad_token_id)[1][0]
        except:
            idx=-1
        inputs['input_ids'][0][idx]=self.tokenizer.mask_token_id
        print(inputs['input_ids'])
        print(idx)
        inputs.to('cuda')

        # Get model outputs, including attention weights
        outputs = self.model(**inputs)
        output_movie = torch.topk(outputs.logits[0][idx], 100)
        ids_to_search = [self.map_asin.loc[id]['asin'] for id in output_movie.indices.cpu().detach().numpy().tolist()]
        result = collection.find({'parent_asin': {'$in': ids_to_search}})
        return [r for r in result]
    
    def get_history(self, user_id):
        asin_history = reviews.find({'user_id': user_id})
        history = []
        for asin in asin_history:
            print(asin)
            print(asin['parent_asin'])
            history.append(asin['parent_asin'])
        return history

    def perform_search(self):
        """Perform a search in MongoDB using full-text search."""
        keyword = self.search_entry.get()
        if keyword:
            self.search_results = list(
                collection.find(
                    {"$text": {"$search": keyword}}
                ).sort([("score", {"$meta": "textScore"})]).limit(100)
            )
            self.showing_default = False
            self.current_page = 0
            self.display_items(self.search_results)
        else:
            self.search_results = []
            self.display_items([])

    def reset_recommender(self):
        reviews.delete_many({'user_id': self.user_id})

    def show_home(self):
        """Show the default items."""
        self.current_page = 0
        self.showing_default = True
        self.user_history = self.get_history(self.user_id)
        self.recommended_items = self.get_recommended()
        self.display_items(self.recommended_items)

    def display_items(self, items):
        """Display items on the current page."""
        # Clear previous widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        # If no items, display a message
        if not items:
            label = tk.Label(self.scroll_frame, text="No results found.", font=("Arial", 14))
            label.pack(pady=20)
            return

        # Display items for the current page
        start_index = self.current_page * self.items_per_page
        end_index = start_index + self.items_per_page
        page_items = items[start_index:end_index]

        for idx, item in enumerate(page_items):
            row = idx % 2  # 5 rows
            col = idx // 2  # 2 columns

            # Create a fixed-size frame for each product
            frame = tk.Frame(self.scroll_frame, borderwidth=1, relief="solid", width=400, height=250)
            frame.grid(row=row, column=col, padx=10, pady=10)
            frame.grid_propagate(False)

            # Product Title
            title_label = tk.Label(frame, text=item.get("title", "No Title"), font=("Arial", 12), wraplength=180)
            title_label.pack(pady=5)

            # Product Image
            image_url = item.get("images", [{}])[0].get("large")
            if image_url:
                response = requests.get(image_url)
                img_data = Image.open(BytesIO(response.content))
                img_data = img_data.resize((100, 100), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_data)

                image_label = tk.Label(frame, image=photo)
                image_label.image = photo
                image_label.pack()

            # Product Price
            price_label = tk.Label(frame, text=f"Price: {item.get('price', 'Not Available')}", font=("Arial", 10))
            price_label.pack()

            # Product Rating
            rating_label = tk.Label(
                frame,
                text=f"Rating: {item.get('average_rating', 'No Ratings')} ({item.get('rating_number', 0)} reviews)",
                font=("Arial", 10),
            )
            rating_label.pack()

            # "Buy" Button
            buy_button = ttk.Button(frame, text="Buy", command=lambda i=item: self.simulate_purchase(i))
            buy_button.pack(pady=5)

    def simulate_purchase(self, item):
        """Simulate a purchase of an item and save it as a review."""
        timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
        review_data = {
            "rating": 5.0,
            "title": "",
            "text": "",
            "images": [],
            "asin": item.get("parent_asin", ""),
            "parent_asin": item.get("parent_asin", ""),
            "user_id": self.user_id,
            "timestamp": timestamp,
            "helpful_vote": 0,
            "verified_purchase": True
        }
        reviews.insert_one(review_data)  # Insert the review into the reviews collection
        print(f"Purchased and saved review for item: {item.get('title', 'No Title')}")

    def show_previous_page(self):
        """Show the previous page of search results or default items."""
        if self.current_page > 0:
            self.current_page -= 1
            items = self.recommended_items if self.showing_default else self.search_results
            self.display_items(items)

    def show_next_page(self):
        """Show the next page of search results or default items."""
        items = self.recommended_items if self.showing_default else self.search_results
        if (self.current_page + 1) * self.items_per_page < len(items):
            self.current_page += 1
            self.display_items(items)


class LoginPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Login Page")
        self.user_id = tk.StringVar()

        # Login Form
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(pady=50)

        tk.Label(frame, text="Enter User ID:", font=("Arial", 14)).grid(row=0, column=0, pady=10)
        self.user_id_entry = ttk.Entry(frame, textvariable=self.user_id, font=("Arial", 14))
        self.user_id_entry.grid(row=0, column=1, pady=10)

        self.login_button = ttk.Button(frame, text="Login", command=self.login)
        self.login_button.grid(row=1, column=0, columnspan=2, pady=20)

    def login(self):
        """Handle the login process."""
        user_id = self.user_id.get().strip()
        if user_id:
            self.root.destroy()  # Close the login window
            main_app(user_id)  # Launch the shopping app with the provided user ID
        else:
            tk.messagebox.showerror("Error", "User ID cannot be empty!")


def main_app(user_id):
    """Start the shopping app."""
    root = tk.Tk()
    ShoppingApp(root, user_id)
    root.mainloop()



def main():
    """Show the login page."""
    root = tk.Tk()
    LoginPage(root)
    root.mainloop()


if __name__ == "__main__":
    main()